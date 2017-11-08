from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import torch.optim as optim
from six.moves import cPickle
from torch.autograd import Variable

import eval_utils_forCommNet
import misc.utils as utils
from dataloader import *
from logger import Logger
from misc.rewards import get_self_critical_reward_forCommNet
from models.CommNetModel import CommNetModel, ShowTellModel
from opts import opts_withCommNet


def update_lr(opt, epoch, model, optimizer_G):

    # Assign the learning rate
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
        utils.set_lr(optimizer_G, opt.current_lr)  # set the decayed rate
    else:
        opt.current_lr = opt.learning_rate
    # Assign the scheduled sampling prob
    if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob

    # If start self critical training
    if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
        sc_flag = True
    else:
        sc_flag = False

    update_lr_flag = False
    return opt, sc_flag, update_lr_flag, model, optimizer_G

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from_S is not None:
        if os.path.isfile(os.path.join(opt.start_from_S, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from_S, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # Set CommNetModel
    model1 = ShowTellModel(opt)
    model2 = ShowTellModel(opt)
    model1.load_state_dict(torch.load(os.path.join(opt.start_from_T, 'model.pth')))
    model2.load_state_dict(torch.load(os.path.join(opt.start_from_S, 'model.pth')))
    model1.cuda()
    model2.cuda()

    model = CommNetModel(opt, model1, model2)
    model.cuda()
    logger = Logger(opt)

    update_lr_flag = True
    model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    # if vars(opt).get('start_from_S', None) is not None and os.path.isfile(os.path.join(opt.start_from_S,"optimizer.pth")):
    #     temp = torch.load(os.path.join(opt.start_from_S, 'optimizer.pth'))
    #     optimizer.load_state_dict(temp)

    while True:
        if update_lr_flag:
            opt, sc_flag, update_lr_flag, model, optimizer = update_lr(opt, epoch, model, optimizer)

        # Load data from train split (0)
        data = loader.get_batch('train', seq_per_img=opt.seq_per_img)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        optimizer.zero_grad()

        if not sc_flag:
            loss_1 = crit(model(fc_feats, att_feats, labels)[0], labels[:,1:], masks[:,1:])
            loss_2 = crit(model(fc_feats, att_feats, labels)[1], labels[:,1:], masks[:,1:])
            loss_1.backward()
            loss_2.backward()
            loss = loss_1 + loss_2
        else:
            gen_result_1, sample_logprobs_1 = model.model1.sample(fc_feats, att_feats, {'sample_max': 0})
            gen_result_2, sample_logprobs_2 = model.model2.sample(fc_feats, att_feats, {'sample_max': 0})

            reward_1 = get_self_critical_reward_forCommNet(model, fc_feats, att_feats, data, gen_result_1, logger)
            reward_2 = get_self_critical_reward_forCommNet(model, fc_feats, att_feats, data, gen_result_2, logger)

            loss_1 = rl_crit(sample_logprobs_1, gen_result_1, Variable(torch.from_numpy(reward_1).float().cuda(), requires_grad=False))
            loss_2 = rl_crit(sample_logprobs_2, gen_result_2, Variable(torch.from_numpy(reward_2).float().cuda(), requires_grad=False))

            loss_1.backward()
            loss_2.backward()
            loss = loss_1 + loss_2

        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()

        if not sc_flag:
            log = "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start)
            logger.write(log)
        else:
            log = "iter {} (epoch {}), S_avg_reward = {:.3f}, T_avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration,  epoch, np.mean(reward_1[:,0]), np.mean(reward_2[:,0]), end - start)
            logger.write(log)

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tf_summary_writer, 'avg_reward_S', np.mean(reward_1[:,0]), iteration)
                    add_summary_value(tf_summary_writer, 'avg_reward_T', np.mean(reward_2[:,0]), iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward_1[:,0]+reward_2[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))

            #val_loss, predictions, lang_stats = eval_utils.eval_split(model.model1, crit, loader, logger, eval_kwargs)
            val_loss, predictions, lang_stats = eval_utils_forCommNet.eval_split(model, crit, loader, logger, eval_kwargs)
            logger.write_dict(lang_stats)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k,v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model1.pth')
                torch.save(model.model1.state_dict(), checkpoint_path)
                print("model1 saved to {}".format(checkpoint_path))
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model2.pth')
                torch.save(model.model2.state_dict(), checkpoint_path)
                print("model2 saved to {}".format(checkpoint_path))
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model1-best.pth')
                    torch.save(model.model1.state_dict(), checkpoint_path)
                    print("model1 saved to {}".format(checkpoint_path))
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model2-best.pth')
                    torch.save(model.model2.state_dict(), checkpoint_path)
                    print("model2 saved to {}".format(checkpoint_path))
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

torch.cuda.set_device(1)
opt = opts_withCommNet.parse_opt()
train(opt)
