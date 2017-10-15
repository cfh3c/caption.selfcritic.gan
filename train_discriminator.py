from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from torch import optim
import time
from six.moves import cPickle
from models.ShowTellModel import ShowTellModel, Discriminator, Distance
from collections import deque


class Discriminator_trainer():

    def __init__(self, opt):
        #super(Discriminator_trainer, self).__init__(opt)

        self.opt = opt
        self.model_G = ShowTellModel(self.opt)
        self.load_pretrain_model_G()
        self.init_opt()
        self.model_D = Discriminator(self.opt)

        self.data_loader = None
        self.max_length = 16

        self.criterion_D = nn.CrossEntropyLoss(size_average=True)

        """ only update trainable parameters """
        D_parameters = filter(lambda p: p.requires_grad, self.model_D.parameters())
        self.D_optimizer = optim.Adam(D_parameters, lr=self.opt.learning_rate)
        self.iteration = 0

    def load_pretrain_model_G(self):
        self.model_G.load_state_dict(torch.load('save/model_backup/showtell/model.pth'))

    def init_opt(self):
        self.opt.max_epoch = 10
        self.opt.save_checkpoint_every = 1000
        self.opt.start_from = None

    def pretrain_discriminator(self, dataloader):

        for group in self.D_optimizer.param_groups:
            group['lr'] = 0.0001

        self.model_D.cuda()
        self.model_G.cuda()

        while True:
            self.iteration += 1
            data = dataloader.get_batch('train')

            torch.cuda.synchronize()

            tmp = [data['fc_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]

            fc_feats, labels, masks = tmp

            self.model_D.zero_grad()
            self.D_optimizer.zero_grad()

            fc_feats_temp = Variable(fc_feats.data.cpu(), volatile=True).cuda()
            labels = Variable(labels.data.cpu()).cuda()

            sample_res, sample_logprobs = self.model_G.sample(fc_feats_temp, {'sample_max':0}) #640, 16
            greedy_res, greedy_logprobs = self.model_G.sample(fc_feats_temp, {'sample_max':1}) #640, 16
            gt_res = labels[:, 1:] # remove start token(0)

            sample_res_embed = self.model_G.embed(Variable(sample_res))
            greedy_res_embed = self.model_G.embed(Variable(greedy_res))
            gt_res_embed = self.model_G.embed(gt_res)

            f_label = Variable(torch.FloatTensor(data['fc_feats'].shape[0]).cuda())
            r_label = Variable(torch.FloatTensor(data['fc_feats'].shape[0]).cuda())
            f_label.data.fill_(0)
            r_label.data.fill_(1)

            f_D_output = self.model_D(sample_res_embed.detach(), fc_feats.detach())
            f_loss = self.criterion_D(f_D_output, f_label.long())
            f_loss.backward()

            r_D_output = self.model_D(gt_res_embed.detach(), fc_feats.detach())
            r_loss = self.criterion_D(r_D_output, r_label.long())
            r_loss.backward()

            D_loss = f_loss + r_loss
            self.D_optimizer.step()

            torch.cuda.synchronize()

            if self.iteration % 1 == 0:
                print('[%d/%d] Discriminator training..  f_loss : %f,  r_loss : %f , D_loss : %f'
                      %(self.iteration, len(dataloader)/self.opt.batch_size, f_loss.data.cpu().numpy()[0],
                        r_loss.data.cpu().numpy()[0], D_loss.data.cpu().numpy()[0]))

            # make evaluation on validation set, and save model
            #if (self.iteration % self.opt.save_checkpoint_every == 0):
            if (self.iteration % 20 == 0):

                checkpoint_path = os.path.join(self.opt.checkpoint_path, 'model_D.pth')
                torch.save(self.model_D.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(self.opt.checkpoint_path, 'optimizer_D.pth')
                torch.save(self.D_optimizer.state_dict(), optimizer_path)

                best_flag = 1
                if best_flag:
                    checkpoint_path = os.path.join(self.opt.checkpoint_path, 'model_D-best.pth')
                    torch.save(self.model_D.state_dict(), checkpoint_path)
                    print("best model saved to {}".format(checkpoint_path))

    def valid_discriminator(self):
        self.save_discriminator()

    def save_discriminator(self):
        checkpoint_path = os.path.join(self.opt.expr_dir, 'model_D_pretrained.pth')
        torch.save(self.model_D.state_dict(), checkpoint_path)

        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer_D_pretrained.pth')
        torch.save(self.D_optimizer.state_dict(), optimizer_path)


class Euclidean_trainer():

    def __init__(self, opt):

        self.opt = opt
        self.model_G = ShowTellModel(self.opt)
        self.load_pretrain_model_G()
        self.init_opt()
        self.model_E = Distance(self.opt)
        self.load_pretrain_model_E()

        self.data_loader = None
        self.max_length = 16
        self.max_epoch = 20

        self.criterion_E = torch.nn.CosineEmbeddingLoss(margin=-1, size_average=True)

        """ only update trainable parameters """
        E_parameters = filter(lambda p: p.requires_grad, self.model_E.parameters())
        self.E_optimizer = optim.Adam(E_parameters, lr=self.opt.learning_rate)
        self.iteration = 0

    def mseloss(self, input, target, flag=1):
        if flag == 1:
            temp = torch.sum(torch.pow((input - target),2) / input.data.shape[1])
        elif flag == -1:
            temp = torch.sum
        else:
            raise Exception, 'flag must be 1 or -1.'

    def NCE_loss(self, input, target, flag=1):
        input, target = F.logsigmoid(input), F.logsigmoid(target)
        G_theta = input-target
        h_theta = 1/(1+torch.exp(-G_theta))
        if flag==1:
            loss = -torch.log(h_theta)
        elif flag==-1:
            loss = -torch.log(1-h_theta)
        else:
            raise Exception, 'flag must be 1 or -1.'
        loss = torch.mean(loss, 1)
        return loss.mean(0)

    def load_pretrain_model_E(self):
        self.model_E.load_state_dict(torch.load('save/model_E_NCE/model_E_10epoch.pth'))

    def load_pretrain_model_G(self):
        self.model_G.load_state_dict(torch.load('save/model_backup/showtell/model.pth'))

    def init_opt(self):
        self.opt.max_epoch = 10
        self.opt.save_checkpoint_every = 1000
        self.opt.start_from = None

    def pretrain_Euclidean(self, dataloader):

        for group in self.E_optimizer.param_groups:
            group['lr'] = 0.001

        self.model_E.cuda()
        self.model_G.cuda()
        self.criterion_E.cuda()

        while True:
            self.iteration += 1
            data = dataloader.get_batch('train')
            tmp = [data['fc_feats'], data['labels'], data['masks']]
            torch.cuda.synchronize()

            self.model_E.zero_grad()
            self.E_optimizer.zero_grad()

            # flag 1 training
            tmp1 = [Variable(torch.from_numpy(_), requires_grad=True).cuda() for _ in tmp]
            fc_feats, labels, masks = tmp1
            batch_size = labels.size(0)
            labels = Variable(labels.data.cpu()).cuda()
            gt_res = labels[:, 1:] # remove start token(0)
            gt_res_embed = self.model_G.embed(gt_res)

            gt_im_output, gt_sent_output = self.model_E(gt_res_embed, fc_feats)
            #loss = self.mseloss(gt_im_output, gt_sent_output)
            #flags = Variable(torch.ones(batch_size)).cuda()
            #loss1 = self.criterion_E(gt_im_output, gt_sent_output, flags)
            loss1 = self.NCE_loss(gt_im_output, gt_sent_output, flag=1)

            # flag -1 training
            tmp2 = self.shuffle_data(tmp)
            tmp2 = [Variable(torch.from_numpy(_), requires_grad=True).cuda() for _ in tmp2]
            fc_feats, labels, masks = tmp2
            batch_size = labels.size(0)
            labels = Variable(labels.data.cpu()).cuda()
            gt_res = labels[:, 1:] # remove start token(0)
            gt_res_embed = self.model_G.embed(gt_res)

            gt_im_output, gt_sent_output = self.model_E(gt_res_embed, fc_feats)
            #loss = self.mseloss(gt_im_output, gt_sent_output)
            #flags = -Variable(torch.ones(batch_size)).cuda()
            #loss2 = self.criterion_E(gt_im_output, gt_sent_output, flags)
            loss2 = self.NCE_loss(gt_im_output, gt_sent_output, flag=-1)

            loss = (loss1 + loss2)
            #loss = loss1
            loss.backward()

            self.E_optimizer.step()

            torch.cuda.synchronize()

            if self.iteration % 1 == 0:
                #print('[%d/%d] Euclidean training..  euclidean distance _loss : %f'
                #      %(self.iteration, len(dataloader)/self.opt.batch_size, loss.data.cpu().numpy()[0]))
                print('[%d/%d] Distance training.. loss1 : %f, loss2 : %f, loss_sum = %f'
                       % (self.iteration, len(dataloader) / self.opt.batch_size,
                          loss1.data.cpu().numpy()[0], loss2.data.cpu()[0], loss.data.cpu()[0]))

            # make evaluation on validation set, and save model
            if (self.iteration % 100 == 0):

                checkpoint_path = os.path.join(self.opt.checkpoint_path, 'model_E.pth')
                torch.save(self.model_E.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(self.opt.checkpoint_path, 'optimizer_E.pth')
                torch.save(self.E_optimizer.state_dict(), optimizer_path)

                best_flag = 1
                if best_flag:
                    checkpoint_path = os.path.join(self.opt.checkpoint_path, 'model_E-best.pth')
                    torch.save(self.model_E.state_dict(), checkpoint_path)
                    print("best model saved to {}".format(checkpoint_path))

            if self.iteration >= len(dataloader) * self.max_epoch:
                break

    def valid_discriminator(self):
        self.save_discriminator()

    def save_discriminator(self):
        checkpoint_path = os.path.join(self.opt.expr_dir, 'model_E_pretrained.pth')
        torch.save(self.model_E.state_dict(), checkpoint_path)

        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer_E_pretrained.pth')
        torch.save(self.E_optimizer.state_dict(), optimizer_path)

    def shuffle_data(self, tmp):
        fc_feats, labels, masks = tmp
        labels, masks = deque(labels), deque(masks)

        k=random.randint(1, fc_feats.shape[0]/5-1)
        labels.rotate(5*k)
        masks.rotate(5*k)

        tmp2 = [fc_feats, np.array(labels), np.array(masks)]
        return tmp2

if __name__ == "__main__":
    import opts
    from dataloader import *

    opt = opts.parse_opt()
    opt.use_att = False
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    #trainer = Discriminator_trainer(opt)
    #trainer.pretrain_discriminator(loader)

    trainer = Euclidean_trainer(opt)
    trainer.pretrain_Euclidean(loader)