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


class CommNetModel(nn.Module):
    def __init__(self, opt, model1, model2):
        super(CommNetModel, self).__init__()
        self.opt = opt

        self.model1 = model1
        self.model2 = model2

        # self.Control_1_state = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.Control_1_mean = nn.Linear(opt.rnn_size, opt.rnn_size)
        #
        # self.Control_2_state = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.Control_2_mean = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.state_embedding_0 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.state_embedding_1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.Control_0 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.Control_1 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.ss_prob = 0.0
        self.seq_length = 16

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)

        state1 = self.model1.init_hidden(batch_size)
        state2 = self.model2.init_hidden(batch_size)

        outputs1, outputs2 = list(), list()

        for i in range(seq.size(1)):
            if i == 0:
                xt1 = self.model1.img_embed(fc_feats)
                xt2 = self.model2.img_embed(fc_feats)
            else:
                if self.model1.training and i >= 2 and self.model1.ss_prob > 0.0:  # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.model1.ss_prob
                    if sample_mask.sum() == 0:
                        it1 = seq[:, i - 1].clone()
                        it2 = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it1 = seq[:, i - 1].data.clone()
                        it2 = seq[:, i - 1].data.clone()
                        prob_prev1 = torch.exp(outputs1[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        prob_prev2 = torch.exp(outputs2[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        it1.index_copy_(0, sample_ind, torch.multinomial(prob_prev1, 1).view(-1).index_select(0, sample_ind))
                        it2.index_copy_(0, sample_ind, torch.multinomial(prob_prev2, 1).view(-1).index_select(0, sample_ind))
                        it1 = Variable(it1, requires_grad=False)
                        it2 = Variable(it2, requires_grad=False)
                else:
                    it1 = seq[:, i - 1].clone()
                    it2 = seq[:, i - 1].clone()

                if i >= 2 and seq[:, i - 1].data.sum() == 0:
                    break
                xt1 = self.model1.embed(it1)
                xt2 = self.model2.embed(it2)

            #mean_state = [(state1[i] + state2[i]) / 2 for i in range(2)]
            #state1_comm = self.Controller(state1, mean_state)
            #state2_comm = self.Controller(state2, mean_state)

            state1_comm = self.Controller2(state1, state2)
            state2_comm = self.Controller2(state2, state1)

            output1, state1 = self.model1.core(xt1.unsqueeze(0), state1_comm)
            output2, state2 = self.model2.core(xt2.unsqueeze(0), state2_comm)

            output1 = F.log_softmax(self.model1.logit(self.model1.dropout(output1.squeeze(0))), dim=1)
            outputs1.append(output1)

            output2 = F.log_softmax(self.model2.logit(self.model2.dropout(output2.squeeze(0))), dim=1)
            outputs2.append(output2)

        outputs1 = torch.cat([_.unsqueeze(1) for _ in outputs1[1:]], 1).contiguous()
        outputs2 = torch.cat([_.unsqueeze(1) for _ in outputs2[1:]], 1).contiguous()

        return outputs1, outputs2

    def Controller(self, state, mean_state, rnn_type='LSTM'):

        if rnn_type =='LSTM':

            Cont1_1 = self.Control_1_state(state[0])      # [0]=cell_state / [1]=hidden_state??
            Cont1_2 = self.Control_1_mean(mean_state[0])
            #state_comm_1 = F.tanh(Cont1_1 + Cont1_2)
            state_comm_1 = F.sigmoid(Cont1_1 + Cont1_2)

            Cont2_1 = self.Control_2_state(state[1])
            Cont2_2 = self.Control_2_mean(mean_state[1])
            #state_comm_2 = F.tanh(Cont2_1 + Cont2_2)
            state_comm_2 = F.sigmoid(Cont2_1 + Cont2_2)

            state_comm = (state_comm_1, state_comm_2)

        elif rnn_type == 'RNN':
            pass
        else:
            pass

        return state_comm


    def Controller2(self, state, state2, rnn_type='LSTM'):

        if rnn_type =='LSTM':
            Cont1_1 = self.state_embedding_0(state[0])
            Cont1_2 = self.state_embedding_0(state2[0])
            #state0 = self.Control_0(Cont1_1 + Cont1_2)
            #state0 = self.Control_0(F.tanh(Cont1_1) + F.tanh(Cont1_2))
            state0 = self.Control_0(F.leaky_relu(Cont1_1) + F.leaky_relu(Cont1_2))

            Cont2_1 = self.state_embedding_1(state[1])
            Cont2_2 = self.state_embedding_1(state2[1])
            #state1 = self.Control_1(Cont2_1 + Cont2_2)
            #state1 = self.Control_1(F.tanh(Cont2_1) + F.tanh(Cont2_2))
            state1 = self.Control_1(F.leaky_relu(Cont2_1) + F.leaky_relu(Cont2_2))

            state_comm = (F.tanh(state0), F.tanh(state1))

        elif rnn_type == 'RNN':
            pass
        else:
            pass

        return state_comm


    def sample_2model(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            raise NameError, 'Fuck you baby'

        batch_size = fc_feats.size(0)
        state1 = self.model1.init_hidden(batch_size)
        state2 = self.model2.init_hidden(batch_size)

        seq1, seq2 = [], []
        seqLogprobs1, seqLogprobs2 = [], []

        for t in range(self.seq_length + 2):
            if t == 0:
                xt1 = self.model1.img_embed(fc_feats)
                xt2 = self.model2.img_embed(fc_feats)
            else:
                if t == 1:  # input <bos>
                    it1 = fc_feats.data.new(batch_size).long().zero_()
                    it2 = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs1, it1 = torch.max(logprobs.data, 1)
                    it1 = it1.view(-1).long()
                    sampleLogprobs2, it2 = torch.max(logprobs.data, 1)
                    it2 = it2.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev1 = torch.exp(logprobs1.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                        prob_prev2 = torch.exp(logprobs2.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev1 = torch.exp(torch.div(logprobs1.data, temperature)).cpu()
                        prob_prev2 = torch.exp(torch.div(logprobs2.data, temperature)).cpu()

                    it1 = torch.multinomial(prob_prev1, 1).cuda()
                    it2 = torch.multinomial(prob_prev2, 1).cuda()
                    sampleLogprobs1 = logprobs1.gather(1, Variable(it1, requires_grad=False))  # gather the logprobs at sampled positions

                    it1 = it1.view(-1).long()  # and flatten indices for downstream processing
                    it2 = it2.view(-1).long()  # and flatten indices for downstream processing

                xt1 = self.model1.embed(Variable(it1, requires_grad=False))
                xt2 = self.model2.embed(Variable(it2, requires_grad=False))

            if t >= 2:
                # stop when all finished
                if t == 2:
                    unfinished = it1 > 0
                else:
                    unfinished = unfinished * (it1 > 0)

                it1 = it1 * unfinished.type_as(it1)
                seq1.append(it1)  # seq[t] the input of t+2 time step
                seqLogprobs1.append(sampleLogprobs1.view(-1))

            mean_state = [(state1[i] + state2[i])/2 for i in range(2)]

            state1_comm = self.Controller(state1, mean_state)
            state2_comm = self.Controller(state2, mean_state)

            output1, state1 = self.model1.core(xt1.unsqueeze(0), state1_comm)
            output2, state2 = self.model2.core(xt2.unsqueeze(0), state2_comm)

            logprobs1 = F.log_softmax(self.model1.logit(self.model1.dropout(output1.squeeze(0))))
            logprobs2 = F.log_softmax(self.model2.logit(self.model2.dropout(output2.squeeze(0))))

            logprobs = logprobs1 + logprobs2

        return torch.cat([_.unsqueeze(1) for _ in seq1], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs1], 1)


    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            raise NameError, 'Fuck you baby'

        batch_size = fc_feats.size(0)
        state1 = self.model1.init_hidden(batch_size)

        seq1 = []
        seqLogprobs1 = []

        for t in range(self.seq_length + 2):
            if t == 0:
                xt1 = self.model1.img_embed(fc_feats)
            else:
                if t == 1:  # input <bos>
                    it1 = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs1, it1 = torch.max(logprobs.data, 1)
                    it1 = it1.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev1 = torch.exp(logprobs1.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev1 = torch.exp(torch.div(logprobs1.data, temperature)).cpu()

                    it1 = torch.multinomial(prob_prev1, 1).cuda()
                    sampleLogprobs1 = logprobs1.gather(1, Variable(it1, requires_grad=False))  # gather the logprobs at sampled positions
                    it1 = it1.view(-1).long()  # and flatten indices for downstream processing

                xt1 = self.model1.embed(Variable(it1, requires_grad=False))

            if t >= 2:
                # stop when all finished
                if t == 2:
                    unfinished = it1 > 0
                else:
                    unfinished = unfinished * (it1 > 0)

                it1 = it1 * unfinished.type_as(it1)
                seq1.append(it1)  # seq[t] the input of t+2 time step
                seqLogprobs1.append(sampleLogprobs1.view(-1))

            #mean_state = [(state1[i])/1 for i in range(2)]
            #state1_comm = self.Controller(state1, mean_state)

            mean_state = [(state1[i])/1 for i in range(2)]
            state1_comm = self.Controller2(state1, mean_state)

            output1, state1 = self.model1.core(xt1.unsqueeze(0), state1_comm)

            logprobs1 = F.log_softmax(self.model1.logit(self.model1.dropout(output1.squeeze(0))))
            logprobs = logprobs1

        return torch.cat([_.unsqueeze(1) for _ in seq1], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs1], 1)



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

    def forward(self, fc_feats, att_feats, seq):
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

    def sample_beam(self, fc_feats, att_feats, opt={}):
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

    def sample(self, fc_feats, att_feats, opt={}):
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