from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    norm = norm.unsqueeze(1)
    X = torch.div(X, norm.expand_as(X))
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score

class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out



class VSE(nn.Module):
    def __init__(self, opt):
        super(VSE, self).__init__()
        self.opt = opt

        self.im_embedding = nn.Linear(2048, 1024)
        self.rnn = nn.GRU(300, 1024, 1, batch_first=True)
        self.embed = nn.Embedding(9487 + 1, 300)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def VSE_sent_embedding(self, embed_sent, lengths, fc_feats):
        packed = pack_padded_sequence(embed_sent, lengths, batch_first=True)
        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(embed_sent.size(0), 1, 1024)-1).cuda()
        out = torch.gather(padded[0], 1, I)
        out = out.squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)
        return out


    def VSE_sent_embedding_test(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, 1024)-1).cuda()
        out = torch.gather(padded[0], 1, I)
        out = out.squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        return out

    def VSE_sent_embedding_test2(self, embed_sent, seq, fc_feats):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i - 1].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i - 1].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind,
                                   torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i - 1].clone()
                # break if all the sequences end
            if i >= 2 and seq[:, i - 1].data.sum() == 0:
                break
            xt = self.embed(it)

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
            outputs.append(output)

        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, 1024)-1).cuda()
        out = torch.gather(padded[0], 1, I)
        out = out.squeeze(1)
        # normalization in the joint embedding space
        out = l2norm(out)

        return out


    def forward(self, embed_sent, lenghts, fc_feats):

        #sent_embedding = self.VSE_sent_embedding(embed_sent, lenghts, fc_feats)
        sent_embedding = self.VSE_sent_embedding_test(embed_sent, lenghts)

        img_embedding = self.im_embedding(fc_feats)
        img_embedding = l2norm(img_embedding)

        return img_embedding, sent_embedding

