from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

CiderD_scorer = CiderD(df='coco-train-idxs')
from collections import deque
import random

#CiderD_scorer = CiderD(df='corpus')

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

#def get_self_critical_reward(model, fc_feats, att_feats, data, gen_result):
def get_self_critical_reward(model, fc_feats ,att_feats, data, gen_result, logger):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    
    # get greedy decoding baseline
    greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
    #greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True))
    #gen_result, sample_logprobs = model.sample(fc_feats, {'sample_max': 0})

    res = OrderedDict()
    
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    logger.write(log)

    scores_sample = scores[:batch_size]
    scores_greedy = scores[batch_size:]

    scores = scores_sample - scores_greedy

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_self_critical_reward_forTS(model, model2, fc_feats, att_feats, data, gen_result, logger):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
    greedy_res_2, _ = model2.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
        res[batch_size + i] = [array_to_str(greedy_res[i])]
        res[batch_size*2 + i] = [array_to_str(greedy_res_2[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(3 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(3 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    logger.write(log)

    scores_sample = scores[:batch_size]
    scores_greedy = scores[batch_size:batch_size*2]
    scores_greedy_2 = scores[batch_size*2:]

    scores_1 = scores_sample - scores_greedy # TS
    scores_2 = scores_sample - scores_greedy_2 # self

    scores = scores_1 * 0.7 + scores_2 * 0.3
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_reward_test(model, fc_feats, data, gen_result, logger):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    # greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
    greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True))
    gen_result, sample_logprobs = model.sample(fc_feats, {'sample_max': 0})
    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    for i in range(batch_size):
        res[batch_size*2 + i] = [array_to_str(data['gts'][i//seq_per_img][i % seq_per_img])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    # _, scores = Bleu(4).compute_score(gts, res)
    # scores = np.array(scores[3])
    res = [{'image_id': i, 'caption': res[i]} for i in range(3 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(3 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    #logger.write(log)

    scores_sample = scores[:batch_size]
    scores_greedy = scores[batch_size:batch_size*2]
    scores_gt = scores[batch_size*2:]

    scores_s_greedy = scores_sample - scores_greedy
    scores_s_gt = scores_sample-scores_gt

    rewards = np.repeat(scores_s_greedy[:, np.newaxis], gen_result.shape[1], 1)

    return rewards



def get_gan_reward(model_G, model_D, criterion_D, fc_feats, data, logger):
    batch_size = fc_feats.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    sample_res, sample_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 0})  # 640, 16
    greedy_res, greedy_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 1})  # 640, 16

    sample_res_embed = model_G.embed(Variable(sample_res, volatile=True)).cuda()
    greedy_res_embed = model_G.embed(Variable(greedy_res, volatile=True)).cuda()

    f_label = Variable(torch.FloatTensor(data['fc_feats'].shape[0]).cuda())
    f_label.data.fill_(0)

    f_D_sample_output = model_D(sample_res_embed.detach(), fc_feats.detach())
    f_D_greedy_output = model_D(greedy_res_embed.detach(), fc_feats.detach())

    # f_sample_loss = criterion_D(f_D_sample_output, f_label.long())
    # f_greedy_loss = criterion_D(f_D_greedy_output, f_label.long())
    # scores = f_sample_loss - f_greedy_loss
    # scores = scores.data.cpu().numpy()

    f_sample_loss = LossForEachBatch(f_D_sample_output, f_label, mode='NLL')
    f_greedy_loss = LossForEachBatch(f_D_greedy_output, f_label, mode='NLL')
    scores = f_sample_loss - f_greedy_loss
    log = 'GAN mean scores: %f' % scores.mean()
    logger.write(log)

    rewards = np.repeat(scores[:, np.newaxis], sample_res.size(1), 1)

    return rewards


def get_distance_reward(model_G, model_E, criterion_E, fc_feats, data, logger, is_mismatched=False):
    batch_size = fc_feats.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    sample_res, sample_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 0})  # 640, 16
    greedy_res, greedy_logprobs = model_G.sample(Variable(fc_feats.data, volatile=True), {'sample_max': 1})  # 640, 16

    sample_res_embed = model_G.embed(Variable(sample_res, volatile=True)).cuda()
    greedy_res_embed = model_G.embed(Variable(greedy_res, volatile=True)).cuda()

    if is_mismatched == True:
        fc_feats = rotate_data(fc_feats)
        flags_sample = -Variable(torch.ones(batch_size)).cuda()
        flags_greedy = -Variable(torch.ones(batch_size)).cuda()
    else:
        flags_sample = Variable(torch.ones(batch_size)).cuda()
        flags_greedy = Variable(torch.ones(batch_size)).cuda()

    f_E_sample_output_im, f_E_sample_output_sent = model_E(sample_res_embed.detach(), fc_feats.detach())
    f_E_greedy_output_im, f_E_greedy_output_sent = model_E(greedy_res_embed.detach(), fc_feats.detach())

    f_sample_distance_loss = CosineLossForEachBatch(f_E_sample_output_im, f_E_sample_output_sent, flags_sample, criterion_E, mode='cosine')
    f_greedy_distance_loss = CosineLossForEachBatch(f_E_greedy_output_im, f_E_greedy_output_sent, flags_greedy, criterion_E, mode='cosine')

    #f_sample_distance = CosineDistanceForEachBatch(f_E_sample_output_im, f_E_sample_output_sent, criterion_E, mode='cosine')
    #f_greedy_distance = CosineDistanceForEachBatch(f_E_greedy_output_im, f_E_greedy_output_sent, criterion_E, mode='cosine')

    scores = f_sample_distance_loss - f_greedy_distance_loss
    #scores = f_sample_distance - f_greedy_distance
    log = 'Distance loss: %f' % scores.mean()
    logger.write(log)

    rewards = np.repeat(scores[:, np.newaxis], sample_res.size(1), 1)
    return rewards

def LossForEachBatch(outputs, labels, mode):
    if mode == 'BCE':
        # output must have passed through F.SIGMOID
        # loss = [-(torch.log(outputs[i]) * (labels[i]) + torch.log(1 - outputs[i]) * (1 - labels[i]))
        #         for i in range(outputs.size(0))]
        # return loss
        pass
    elif mode == 'NLL':
        outputs = F.softmax(outputs)
        #labels = Variable(torch.FloatTensor(labels.data.cpu().numpy())).cuda()
        labels = labels.cuda()

        loss = [-(torch.log(outputs[i][1]) * labels[i] + torch.log(outputs[i][0]) * (1 - labels[i])).data.cpu().numpy()[0]
                for i in range(outputs.size(0))]
        return np.array(loss)

    elif mode == 'Acc':
        # loss = [outputs[i][1] for i in range(outputs.size(0))]
        # return loss
        pass
    else:
        raise Exception('mode options must be BCE or NLL.')

def CosineLossForEachBatch(outputs_im, outputs_sent, flags, criterion, mode):
    if mode == '.':
        pass
    elif mode == 'cosine':
        loss = [criterion(outputs_im[i].unsqueeze(0), outputs_sent[i].unsqueeze(0), flags[i].unsqueeze(0)).data.cpu().numpy()[0]
                for i in range(outputs_im.size(0))]
        return np.array(loss)
    else:
        raise Exception('mode options check plz.')

def CosineDistanceForEachBatch(outputs_im, outputs_sent, cosfunc, mode):
    if mode == '.':
        pass
    elif mode == 'cosine':
        distance = [cosfunc(outputs_im[i].unsqueeze(0), outputs_sent[i].unsqueeze(0)).data.cpu().numpy()[0]
                for i in range(outputs_im.size(0))]
        return np.array(distance)
    else:
        raise Exception('mode options check plz.')

def rotate_data(fc_feats, is_cuda=True):
    k = random.randint(1, fc_feats.size()[0] / 5 - 1)
    fc_feats = fc_feats.data.cpu().numpy()
    tmp = deque(fc_feats)
    tmp.rotate(5 * k)
    tmp = Variable(torch.FloatTensor(np.array(tmp)))
    if is_cuda:
        tmp = tmp.cuda()
    return tmp



def get_self_critical_reward_forTS_diff(model, model2, fc_feats, att_feats, data, gen_result, logger):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
    greedy_res_2, _ = model2.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
        res[batch_size + i] = [array_to_str(greedy_res[i])]
        res[batch_size*2 + i] = [array_to_str(greedy_res_2[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(3 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(3 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    logger.write(log)

    scores_sample = scores[:batch_size]
    scores_greedy = scores[batch_size:batch_size*2]
    scores_greedy_2 = scores[batch_size*2:]

    scores_1 = scores_sample - scores_greedy # TS
    scores_2 = scores_sample - scores_greedy_2 # self

    scores = scores_1 * 0.7 + scores_2 * 0.3
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_self_critical_reward_forCommNet(model,fc_feats, att_feats, data, gen_result, logger, mode='None'):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    if mode == '1':
        greedy_res, _ = model.model1.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
        greedy_res_2, _ = model.model2.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
    elif mode == '2':
        greedy_res, _ = model.model2.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
        greedy_res_2, _ = model.model1.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))
    else:
        raise NameError, 'plz check the mode : must be 1 or 2'

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
        res[batch_size + i] = [array_to_str(greedy_res[i])]
        res[batch_size*2 + i] = [array_to_str(greedy_res_2[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(3 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(3 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    logger.write(log)

    scores_sample = scores[:batch_size]
    scores_greedy = scores[batch_size:batch_size*2]
    scores_greedy_2 = scores[batch_size*2:]

    scores_1 = scores_sample - scores_greedy # TS
    scores_2 = scores_sample - scores_greedy_2 # self

    scores = scores_1 * 0.7 + scores_2 * 0.3
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_self_critical_reward_forFNet(model, fc_feats, att_feats, data, gen_result, gen_result_2, logger):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    greedy_res, greedy_res_2, _, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True), mode='sc')

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    gen_result_2 = gen_result_2.cpu().numpy()

    greedy_res = greedy_res.cpu().numpy()
    greedy_res_2 = greedy_res_2.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
        res[batch_size + i] = [array_to_str(gen_result_2[i])]
        res[batch_size*2 + i] = [array_to_str(greedy_res[i])]
        res[batch_size*3 + i] = [array_to_str(greedy_res_2[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(4 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(4 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    logger.write(log)

    scores_sample = scores[:batch_size]
    scores_sample_2 = scores[batch_size:batch_size*2]
    scores_greedy = scores[batch_size*2:batch_size*3]
    scores_greedy_2 = scores[batch_size*3:]

    scores_1 = scores_sample - scores_greedy
    scores_2 = scores_sample_2 - scores_greedy_2

    rewards_1 = np.repeat(scores_1[:, np.newaxis], gen_result.shape[1], 1)
    rewards_2 = np.repeat(scores_2[:, np.newaxis], gen_result_2.shape[1], 1)

    return rewards_1, rewards_2


def get_self_critical_reward_forSeriNet(model, fc_feats, att_feats, data, gen_result, gen_result_2, logger):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    greedy_res, _, greedy_res_2, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True), mode='sc')

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    gen_result_2 = gen_result_2.cpu().numpy()

    greedy_res = greedy_res.cpu().numpy()
    greedy_res_2 = greedy_res_2.cpu().numpy()

    #test1, test2, test3, test4 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
        res[batch_size + i] = [array_to_str(gen_result_2[i])]
        res[batch_size*2 + i] = [array_to_str(greedy_res[i])]
        res[batch_size*3 + i] = [array_to_str(greedy_res_2[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(4 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(4 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    log = 'Cider scores:' + str(_)
    logger.write(log)

    scores_sample = scores[:batch_size]
    scores_sample_2 = scores[batch_size:batch_size*2]
    scores_greedy = scores[batch_size*2:batch_size*3]
    scores_greedy_2 = scores[batch_size*3:]

    scores_2 = scores_sample_2 - scores_greedy_2
    #scores_2 = scores_sample_2 - (scores_greedy + scores_greedy_2)/2

    rewards_2 = np.repeat(scores_2[:, np.newaxis], gen_result_2.shape[1], 1)

    return rewards_2