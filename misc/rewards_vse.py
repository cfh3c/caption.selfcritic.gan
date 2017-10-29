from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time
from collections import OrderedDict

import numpy
import numpy as np
import torch
from torch.autograd import Variable

sys.path.append("cider")

from collections import deque
import random
from models.VSE import order_sim
import torch.nn as nn


def get_currscore_reward(model_G, model_VSE, fc_feats, gen_result, logger):
    batch_size = fc_feats.size(0)  # batch_size = sample_size * seq_per_img

    #sample_res, sample_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 0})  # 640, 16
    greedy_res, greedy_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 1})  # 640, 16

    rewards_sample, sample_masks = validate_curr(model_VSE, fc_feats, gen_result)
    rewards_greedy, greedy_masks = validate_curr(model_VSE, fc_feats, greedy_res)

    rewards = -(rewards_sample - rewards_greedy) + 0.0
    log = 'currscore mean rewards: %f' % rewards.mean()
    logger.write(log)

    rewards_sample = np.repeat(rewards_sample[:, np.newaxis], sample_masks.shape[1], 1)
    #rewards_sample = rewards_sample * sample_masks
    rewards_greedy = np.repeat(rewards_greedy[:, np.newaxis], greedy_masks.shape[1], 1)
    #rewards_greedy = rewards_greedy * greedy_masks

    #rewards = -(rewards_sample - rewards_greedy)
    #reward_bl = np.repeat(np.repeat(reward_bl, masks.shape[0])[:,np.newaxis],masks.shape[1],1)
    #rewards = np.repeat(rewards[:, np.newaxis], sample_masks.shape[1], 1)
    #rewards = (rewards - reward_bl) * masks

    # CURRICULUM
    #decay = np.linspace(0, 1, num=sample_masks.shape[1])
    #decay = np.repeat(decay[np.newaxis,:], greedy_masks.shape[0], 0)
    #rewards = rewards * decay

    # LOG REWARD
    rewards_sample = 1/(1+np.exp(-np.log(rewards_sample + 1e-8)))
    rewards_greedy = 1/(1+np.exp(-np.log(rewards_greedy + 1e-8)))
    rewards = -(rewards_sample - rewards_greedy)

    return rewards

def get_sim_reward(model_G, model_VSE, fc_feats, gen_result, logger):
    batch_size = fc_feats.size(0)  # batch_size = sample_size * seq_per_img

    sample_res, sample_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 0})  # 640, 16
    #greedy_res, greedy_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 1})  # 640, 16

    rewards_sample, sample_masks = validate_sim(model_VSE, fc_feats, sample_res)
    rewards_greedy, greedy_masks = validate_sim(model_VSE, fc_feats, gen_result)

    rewards = rewards_sample - rewards_greedy
    log = 'currscore mean rewards: %f' % rewards.mean()
    logger.write(log)

    rewards_sample = np.repeat(rewards_sample[:, np.newaxis], sample_masks.shape[1], 1)
    rewards_sample = rewards_sample# * sample_masks
    rewards_greedy = np.repeat(rewards_greedy[:, np.newaxis], greedy_masks.shape[1], 1)
    rewards_greedy = rewards_greedy# * greedy_masks

    #rewards = rewards_sample - rewards_greedy

    rewards_sample = 1/(1+np.exp(-rewards_sample + 1e-8))
    rewards_greedy = 1/(1+np.exp(-rewards_greedy + 1e-8))
    rewards = -(rewards_sample - rewards_greedy)
    return rewards

def rotate_data(fc_feats, is_cuda=True):
    k = random.randint(1, fc_feats.size()[0] / 5 - 1)
    fc_feats = fc_feats.data.cpu().numpy()
    tmp = deque(fc_feats)
    tmp.rotate(5 * k)
    tmp = Variable(torch.FloatTensor(np.array(tmp)))
    if is_cuda:
        tmp = tmp.cuda()
    return tmp


def reordering_data(data):
    lengths = np.array([s.sum() - 1 - 1 for s in data['masks']]).astype('int')
    idx = np.argsort(-lengths)
    fc_feats = [data['fc_feats'][i] for i in idx]
    labels = [data['labels'][i][1:] for i in idx]
    #masks = [data['masks'][i] for i in idx]
    lengths = [lengths[i] for i in idx]

    fc_feats = torch.from_numpy(np.array(fc_feats))
    labels = torch.from_numpy(np.array(labels))
    lengths = np.array(lengths)

    return [fc_feats, labels, lengths]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def reordering_batch_data(fc_feats, gen_result):
    mask = gen_result > 0
    fc_feats, gen_result, mask = fc_feats.data.cpu().numpy(), gen_result.cpu().numpy(), mask.cpu().numpy()
    lengths = np.array([max(s.sum()-1,1) for s in mask]).astype('int')

    idx = np.argsort(-lengths)

    fc_feats = [fc_feats[i] for i in idx]
    labels = [gen_result[i][1:] for i in idx]
    lengths = [lengths[i] for i in idx]
    mask = [mask[i] for i in idx]

    fc_feats = torch.from_numpy(np.array(fc_feats))
    labels = torch.from_numpy(np.array(labels))
    lengths = np.array(lengths)
    mask = np.array(mask)

    return [fc_feats, labels, lengths], mask, idx


def encode_data(model, fc_feats, gen_result):
    """Encode all images and captions loadable by `data_loader`
    """

    batch_size = fc_feats.size(0)

    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()
    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None

    val_data, masks, reordered_idx = reordering_batch_data(fc_feats, gen_result)
    images, captions, lengths = val_data
    ids = range(batch_size)
    # compute the embeddings
    img_emb, cap_emb = model.forward_emb(images, captions, lengths, volatile=True)

    # initialize the numpy arrays given the size of the embeddings
    if img_embs is None:
        img_embs = np.zeros((batch_size, img_emb.size(1)))
        cap_embs = np.zeros((batch_size, cap_emb.size(1)))

    # preserve the embeddings by copying from gpu and converting to numpy
    img_embs[ids] = img_emb.data.cpu().numpy().copy()
    cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

    # measure accuracy and record loss
    model.forward_loss(img_emb, cap_emb)

    # measure elapsed time
    batch_time.update(time.time() - end)

    # if i % 1 == 0:
    #     print('Test: [{0}/{1}]\t'
    #             '{e_log}\t'
    #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #             .format(
    #                 i*len(images), len(data_loader.split_ix['val']), batch_time=batch_time,
    #                 e_log=str(model.logger)))
    #del images, captions

    #return img_embs, cap_embs, masks

    rereordered_img_embeds = np.zeros(img_embs.shape)
    rereordered_cap_embeds = np.zeros(cap_embs.shape)
    rereordered_mask = np.zeros(masks.shape)
    for i, idx in enumerate(reordered_idx):
        rereordered_img_embeds[idx] = img_embs[i]
        rereordered_cap_embeds[idx] = cap_embs[i]
        rereordered_mask[idx] = masks[i]

    return rereordered_img_embeds, rereordered_cap_embeds, rereordered_mask



def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    k=1 # k=1 or k=5
    if npts is None:
        npts = int(images.shape[0] / k)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[k * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], k * (index + bs))
                im2 = images[k * index:mx:k]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(k * index, k * index + k, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    #r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    #r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    #r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)

    rank_reward = 1.0 * ranks / images.shape[0]

    r1 = len(numpy.where(ranks < 1)[0]) * 100. / len(ranks)
    r3 = len(numpy.where(ranks < 5)[0]) * 100. / len(ranks)
    r5 = len(numpy.where(ranks < 10)[0]) * 100. / len(ranks)

    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r3, r5, medr, meanr), (ranks, top1)
    else:
        return (r1, r3, r5, medr, meanr), rank_reward


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    k=1 # k=1 or 5
    if npts is None:
        npts = int(images.shape[0] / k)
    ims = numpy.array([images[i] for i in range(0, len(images), k)])

    ranks = numpy.zeros(k * npts)
    top1 = numpy.zeros(k * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[k * index:k * index + k]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], k * index + bs)
                q2 = captions[k * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (k * index) % bs:(k * index) % bs + k].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[k * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[k * index + i] = inds[i][0]

    # Compute metrics
    #r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    #r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    #r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)

    rank_reward = 1.0 * ranks / images.shape[0]

    r1 = len(numpy.where(ranks < 1)[0]) * 100. / len(ranks)
    r3 = len(numpy.where(ranks < 5)[0]) * 100. / len(ranks)
    r5 = len(numpy.where(ranks < 10)[0]) * 100. / len(ranks)

    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r3, r5, medr, meanr), (ranks, top1)
    else:
        return (r1, r3, r5, medr, meanr), rank_reward


def validate_curr(model, fc_feats, gen_result, data_loader=None):
    #val_loader = data_loader
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, masks = encode_data(model, fc_feats, gen_result)

    # caption retrieval
    (r1, r3, r5, medr, meanr), i2t_rewards = i2t(img_embs, cap_embs, measure='cosine')
    #logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r3i, r5i, medri, meanr), t2i_rewards = t2i(img_embs, cap_embs, measure='cosine')
    #logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %  (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    #currscore = (r1 + r3 + r5 + r1i + r3i + r5i)/6 - 0.0

    rewards = t2i_rewards + i2t_rewards
    #reward_bl = np.mean(rewards)

    return rewards, masks#, reward_bl #currscore, masks



def validate_sim(model, fc_feats, gen_result, data_loader=None):

    img_embs, cap_embs, masks = encode_data(model, fc_feats, gen_result)

    img_embs_norm = np.linalg.norm(img_embs, ord=2, axis=1)
    cap_embs_norm = np.linalg.norm(cap_embs, ord=2, axis=1)

    temp = np.sum(img_embs * cap_embs, axis=1)
    temp2 = img_embs_norm * cap_embs_norm
    sim = temp/temp2
    return sim, masks