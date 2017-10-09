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


class ShowTellModel(nn.Module):
    def __init__(self, opt):
        super(ShowTellModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0  # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers,
                                                       bias=False, dropout=self.drop_prob_lm)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

        # self.D_network = getattr(nn, self.rnn_type.upper())(512, 512, 1, bias=False, dropout=self.drop_prob_lm)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def forward(self, fc_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:  # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i - 1].data.clone()
                        # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
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

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def sample_beam(self, fc_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
            for t in range(self.seq_length + 2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    ys, ix = torch.sort(logprobsf, 1,
                                        True)  # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 2:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 2:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t - 2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 2:
                            beam_seq[:t - 2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 2, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 2, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length + 1:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix]
                                                       })

                    # encode as vectors
                    it = beam_seq[t - 2]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 2:
                    state = new_state

                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:  # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, Variable(it,
                                                                 requires_grad=False))  # gather the logprobs at sampled positions
                    it = it.view(-1).long()  # and flatten indices for downstream processing

                xt = self.embed(Variable(it, requires_grad=False))

            if t >= 2:
                # stop when all finished
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0:
                #     break
                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt

        self.net_D = nn.Sequential(nn.Linear(512+512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 2))

        self.bi_lstm = nn.LSTM(input_size=self.opt.D_hidden_size, hidden_size=self.opt.D_hidden_size, bias=True,
                               batch_first=True, bidirectional=True)

        self.W_conv1 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=16, kernel_size=512))
        self.W_sent_emb1 = nn.Sequential(nn.Linear(1024, 512))

        self.im_embedding = nn.Sequential(nn.Linear(2048, 512),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU())

        self.conv2d_1 = nn.Conv2d(1, 64, (1, 512))
        self.conv2d_2 = nn.Conv2d(1, 64, (3, 512))
        self.conv2d_3 = nn.Conv2d(1, 64, (5, 512))

        self.gmlp1 = nn.Linear(1024, 512)
        self.gmlp2 = nn.Linear(512, 512)
        self.gmlp3 = nn.Linear(512, 512)
        self.fmlp1 = nn.Linear(512, 512)
        self.fmlp2 = nn.Linear(512, 512)

    def self_attentive_sentence_embedding(self, res_embed, fc_feats):
        iter_batch_size = res_embed.size()[0]
        res_embed = torch.transpose(res_embed, 0, 1)  # [16, 640, 512]
        res_embed = res_embed[:16]  # [16, 640, 512]

        bilstm_out, hn = self.bi_lstm(res_embed)  # [16, 640, 1024]

        H_st_ = torch.transpose(bilstm_out, 0, 1)  # [640, 16, 1024]
        H_st = torch.cat(H_st_, 0)  # [640x16,  1024]
        H_st = self.W_sent_emb1(H_st)  # [640x16,  512 ]
        H_st = H_st.view(iter_batch_size, 16, 512)  # [640, 16, 512 ]
        # H_im = self.W_im_emb(im_input)                # [640x16, 512]
        # H_im = H_im.repeat(16, 1, 1).transpose(0, 1)  # [640, 16, 512]
        # H_ = torch.cat((H_st, H_im), 2)  # [640, 16, 1024]

        H_ = H_st
        H_ = self.W_conv1(H_)  # [640, 16, 512]
        H_ = H_.squeeze(2)  # [640, 16]

        attention_ = F.softmax(H_)
        attention = attention_.unsqueeze(2)  # attention = [640, 16, 1]
        attention = attention.repeat(1, 1, 1024)  # attention = [640, 16, 1024]
        embedding = attention * H_st_  # embedding = [640, 16, 1024]
        embedding = torch.sum(embedding, dim=1)  # embedding = [640, 1, 1024]

        return embedding.squeeze(1)  # embedding = [640, 1024]

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def sentence_embedding_my(self, res_embed):

        res_embed = torch.transpose(res_embed, 0, 1)     # [16, 640, 512]
        res_embed = res_embed[:16]                       # [16, 640, 512]

        #bilstm_out, hn = self.bi_lstm(res_embed)         # [16, 640, 1024]
        bilstm_out = torch.transpose(res_embed, 0, 1)   # [640, 16, 1024]
        bilstm_out = bilstm_out.unsqueeze(1)

        x1 = self.conv_and_pool(bilstm_out, self.conv2d_1) #(N,Co)
        x2 = self.conv_and_pool(bilstm_out, self.conv2d_2) #(N,Co)
        x3 = self.conv_and_pool(bilstm_out, self.conv2d_3) #(N,Co)

        feat = torch.cat((x1, x2, x3), 1)  # (N,len(Ks)*Co)

        return feat

    def relation_embedding(self, res_embed):

        batch_size  = res_embed.size(0) # 640
        seq_len     = res_embed.size(1) # 16

        n_object = seq_len              # 16
        n_object_pair = n_object * n_object # 256

        feature_dim = res_embed.size()[-1]  # 512
        out = res_embed.view(batch_size, n_object, feature_dim) # 640, 16, 512

        feature1 = out.unsqueeze(1)
        feature1 = feature1.expand(batch_size, n_object, n_object, feature_dim)
        feature1 = feature1.contiguous().view(-1, n_object_pair, feature_dim) # 640, 256, 512

        feature2 = out.unsqueeze(3)
        feature2 = feature2.expand(batch_size, n_object, feature_dim, n_object)
        feature2 = feature2.contiguous().view(-1, n_object_pair, feature_dim) # 640, 256, 512

        feature = torch.cat([feature1, feature2], 2).view(-1, 512*2)          # 640, 256, 1024

        out = F.relu(self.gmlp1(feature)) # 4, 64, 12288
        out = F.relu(self.gmlp2(out))
        out = F.relu(self.gmlp3(out))

        # Element-wise Sum
        out = out.view(-1, n_object_pair, 512).mean(1)

        return out


    def forward(self, input, im_input):
        #sent_embedding = self.self_attentive_sentence_embedding(input, im_input)
        #sent_embedding = self.sentence_embedding_my(input)
        sent_embedding = self.relation_embedding(input)

        img_embedding = self.im_embedding(im_input)
        embedding = torch.cat((sent_embedding, img_embedding), 1)
        out = self.net_D(embedding)
        return out


class Distance(nn.Module):
    def __init__(self, opt):
        super(Distance, self).__init__()
        self.opt = opt

        self.net_D = nn.Sequential(nn.Linear(512+512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 2))

        self.bi_lstm = nn.LSTM(input_size=self.opt.D_hidden_size, hidden_size=self.opt.D_hidden_size, bias=True,
                               batch_first=True, bidirectional=True)

        self.W_conv1 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=16, kernel_size=512))
        self.W_sent_emb1 = nn.Sequential(nn.Linear(1024, 512))

        self.im_embedding = nn.Sequential(nn.Linear(2048, 512),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU())

        self.conv2d_1 = nn.Conv2d(1, 64, (1, 512))
        self.conv2d_2 = nn.Conv2d(1, 64, (3, 512))
        self.conv2d_3 = nn.Conv2d(1, 64, (5, 512))

        self.gmlp1 = nn.Linear(1024, 512)
        self.gmlp2 = nn.Linear(512, 512)
        self.gmlp3 = nn.Linear(512, 512)
        self.fmlp1 = nn.Linear(512, 512)
        self.fmlp2 = nn.Linear(512, 512)

    def self_attentive_sentence_embedding(self, res_embed, fc_feats):
        iter_batch_size = res_embed.size()[0]
        res_embed = torch.transpose(res_embed, 0, 1)  # [16, 640, 512]
        res_embed = res_embed[:16]  # [16, 640, 512]

        bilstm_out, hn = self.bi_lstm(res_embed)  # [16, 640, 1024]

        H_st_ = torch.transpose(bilstm_out, 0, 1)  # [640, 16, 1024]
        H_st = torch.cat(H_st_, 0)  # [640x16,  1024]
        H_st = self.W_sent_emb1(H_st)  # [640x16,  512 ]
        H_st = H_st.view(iter_batch_size, 16, 512)  # [640, 16, 512 ]
        # H_im = self.W_im_emb(im_input)                # [640x16, 512]
        # H_im = H_im.repeat(16, 1, 1).transpose(0, 1)  # [640, 16, 512]
        # H_ = torch.cat((H_st, H_im), 2)  # [640, 16, 1024]

        H_ = H_st
        H_ = self.W_conv1(H_)  # [640, 16, 512]
        H_ = H_.squeeze(2)  # [640, 16]

        attention_ = F.softmax(H_)
        attention = attention_.unsqueeze(2)  # attention = [640, 16, 1]
        attention = attention.repeat(1, 1, 1024)  # attention = [640, 16, 1024]
        embedding = attention * H_st_  # embedding = [640, 16, 1024]
        embedding = torch.sum(embedding, dim=1)  # embedding = [640, 1, 1024]

        return embedding.squeeze(1)  # embedding = [640, 1024]

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def sentence_embedding_my(self, res_embed):

        res_embed = torch.transpose(res_embed, 0, 1)     # [16, 640, 512]
        res_embed = res_embed[:16]                       # [16, 640, 512]

        #bilstm_out, hn = self.bi_lstm(res_embed)         # [16, 640, 1024]
        bilstm_out = torch.transpose(res_embed, 0, 1)   # [640, 16, 1024]
        bilstm_out = bilstm_out.unsqueeze(1)

        x1 = self.conv_and_pool(bilstm_out, self.conv2d_1) #(N,Co)
        x2 = self.conv_and_pool(bilstm_out, self.conv2d_2) #(N,Co)
        x3 = self.conv_and_pool(bilstm_out, self.conv2d_3) #(N,Co)

        feat = torch.cat((x1, x2, x3), 1)  # (N,len(Ks)*Co)

        return feat

    def relation_embedding(self, res_embed):

        batch_size  = res_embed.size(0) # 640
        seq_len     = res_embed.size(1) # 16

        n_object = seq_len              # 16
        n_object_pair = n_object * n_object # 256

        feature_dim = res_embed.size()[-1]  # 512
        out = res_embed.view(batch_size, n_object, feature_dim) # 640, 16, 512

        feature1 = out.unsqueeze(1)
        feature1 = feature1.expand(batch_size, n_object, n_object, feature_dim)
        feature1 = feature1.contiguous().view(-1, n_object_pair, feature_dim) # 640, 256, 512

        feature2 = out.unsqueeze(3)
        feature2 = feature2.expand(batch_size, n_object, feature_dim, n_object)
        feature2 = feature2.contiguous().view(-1, n_object_pair, feature_dim) # 640, 256, 512

        feature = torch.cat([feature1, feature2], 2).view(-1, 512*2)          # 640, 256, 1024

        out = F.relu(self.gmlp1(feature)) # 4, 64, 12288
        out = F.relu(self.gmlp2(out))
        out = F.relu(self.gmlp3(out))

        # Element-wise Sum
        out = out.view(-1, n_object_pair, 512).mean(1)

        return out


    def forward(self, input, im_input):
        #sent_embedding = self.self_attentive_sentence_embedding(input, im_input)
        #sent_embedding = self.sentence_embedding_my(input)
        sent_embedding = self.relation_embedding(input)
        img_embedding = self.im_embedding(im_input)

        return img_embedding, sent_embedding


def mseloss(input, target):
    temp = torch.sum(torch.pow((input - target), 2) / input.data.shape[1])
    return temp
