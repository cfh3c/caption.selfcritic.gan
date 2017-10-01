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
from models.ShowTellModel import ShowTellModel, Discriminator


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
        self.D_optimizer = optim.Adam(D_parameters, lr=self.opt.learning_rate * 1e-1)
        self.iteration = 0

    def load_pretrain_model_G(self):
        self.model_G.load_state_dict(torch.load('save/model.pth'))

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

            fc_feats = Variable(fc_feats.data.cpu(), volatile=True).cuda()
            labels = Variable(labels.data.cpu()).cuda()

            sample_res, sample_logprobs = self.model_G.sample(fc_feats, {'sample_max':0}) #640, 16
            greedy_res, greedy_logprobs = self.model_G.sample(fc_feats, {'sample_max':1}) #640, 16
            gt_res = labels # 640, 18

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
                print('[%d/%d] Discriminator loss : %f' %(self.iteration, len(dataloader)/128, D_loss.data.cpu().numpy()[0]))

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



if __name__ == "__main__":
    import opts
    from dataloader import *

    opt = opts.parse_opt()
    opt.use_att = False
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    trainer = Discriminator_trainer(opt)
    trainer.pretrain_discriminator(loader)