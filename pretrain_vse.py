from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch import optim
from torch.autograd import *

from models.ShowTellModel import ShowTellModel
from models.VSE import VSE


class Vsepp_trainer():

    def __init__(self, opt):

        self.opt = opt
        self.model_G = ShowTellModel(self.opt)
        self.load_pretrain_model_G()
        self.init_opt()
        self.model_VSE = VSE(self.opt)
        #self.load_pretrain_model_VSE()

        self.data_loader = None
        self.max_length = 16
        self.max_epoch = 20

        """ only update trainable parameters """
        self.VSE_parameters = filter(lambda p: p.requires_grad, self.model_VSE.parameters())
        self.VSE_optimizer = optim.Adam(self.VSE_parameters, lr=self.opt.learning_rate)
        self.iteration = 0

        # Loss and Optimizer
        self.criterion_VSE = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)

        self.Eiters = 0

    def load_pretrain_model_VSE(self):
        self.model_VSE.load_state_dict(torch.load('save/model_E_NCE/.pth'))

    def load_pretrain_model_G(self):
        self.model_G.load_state_dict(torch.load('save/model_backup/showtell/model.pth'))

    def init_opt(self):
        self.opt.max_epoch = 20
        self.opt.save_checkpoint_every = 1000
        self.opt.start_from = None

    def reordering_data(self, tmp):
        fc_feats, labels, masks = tmp
        lengths = np.array([s.sum()-1 for s in masks]).astype('int')
        idx = np.argsort(-lengths)
        fc_feats = [fc_feats[i] for i in idx]
        labels = [labels[i] for i in idx]
        masks = [masks[i] for i in idx]
        lengths = [lengths[i] for i in idx]
        tmp = [np.array(fc_feats), np.array(labels), np.array(masks)]
        return lengths, tmp

    def pretrain_VSE(self, dataloader):

        for group in self.VSE_optimizer.param_groups:
            group['lr'] = 0.0005

        self.model_VSE.cuda()
        self.model_G.cuda()
        self.criterion_VSE.cuda()

        while True:
            self.iteration += 1
            data = dataloader.get_batch('train')
            tmp = [data['fc_feats'], data['labels'], data['masks']]
            lengths, tmp = self.reordering_data(tmp)
            torch.cuda.synchronize()

            self.model_VSE.zero_grad()
            self.VSE_optimizer.zero_grad()

            # flag 1 training
            tmp = [Variable(torch.from_numpy(_), requires_grad=True).cuda() for _ in tmp]
            fc_feats, labels, masks = tmp
            labels = Variable(labels.data.cpu()).cuda()
            gt_res = labels[:, 1:] # remove start token(0)
            #gt_res_embed = self.model_G.embed(gt_res)

            #gt_im_output, gt_sent_output = self.model_VSE(gt_res_embed, lengths, fc_feats)
            gt_im_output, gt_sent_output = self.model_VSE(gt_res, lengths, fc_feats)

            loss = self.criterion_VSE(gt_im_output, gt_sent_output)
            loss.backward()

            if 1:
                self.clip_grad_norm(self.VSE_parameters, 0.2)
            self.VSE_optimizer.step()

            if self.iteration % 1 == 0:
                print('[%d/%d] VSE training.. loss_sum = %f'
                       % (self.iteration, len(dataloader) / self.opt.batch_size, loss.data.cpu().numpy()[0]))

            # make evaluation on validation set, and save model
            if (self.iteration % 100 == 0):

                checkpoint_path = os.path.join(self.opt.checkpoint_path, 'model_VSE.pth')
                torch.save(self.model_VSE.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(self.opt.checkpoint_path, 'optimizer_VSE.pth')
                torch.save(self.VSE_optimizer.state_dict(), optimizer_path)

                best_flag = 1
                if best_flag:
                    checkpoint_path = os.path.join(self.opt.checkpoint_path, 'model_VSE-best.pth')
                    torch.save(self.model_VSE.state_dict(), checkpoint_path)
                    print("best model saved to {}".format(checkpoint_path))

            if self.iteration >= len(dataloader) * self.max_epoch:
                break

    def clip_grad_norm(self, parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        return total_norm

    def valid_discriminator(self):
        self.save_discriminator()

    def save_discriminator(self):
        checkpoint_path = os.path.join(self.opt.expr_dir, 'model_VSE_pretrained.pth')
        torch.save(self.model_VSE.state_dict(), checkpoint_path)

        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer_VSE_pretrained.pth')
        torch.save(self.VSE_optimizer.state_dict(), optimizer_path)

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = self.order_sim
        else:
            self.sim = self.cosine_sim

        self.max_violation = max_violation

    def cosine_sim(self, im, s):
        """Cosine similarity between all the image and sentence pairs
        """
        return im.mm(s.t())

    def order_sim(self, im, s):
        """Order embeddings similarity measure $max(0, s-im)$
        """
        YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
               - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
        score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
        return score

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


if __name__ == "__main__":
    import opts
    from dataloader import *

    opt = opts.parse_opt()
    opt.use_att = False
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    trainer = Vsepp_trainer(opt)
    trainer.pretrain_VSE(loader)
