#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from sklearn import preprocessing
# enc = preprocessing.OneHotEncoder()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.98'
# import paddle
import paddle.fluid as fluid
import numpy as np
import modelsd as models
import time
from data_generater1 import *
# import dataprocess
# from visualdl import LogWriter


def save_model(save_weights_path=''):
    if save_weights_path.endswith('/'):
        fluid.io.save_persistables(
            exe, dirname=save_weights_path, main_program=tp)
    else:
        fluid.io.save_persistables(
            exe, dirname="", filename=save_weights_path, main_program=tp)


def load_premodel(weights_paths=' '):
    if weights_paths.endswith('/'):
        fluid.io.load_params(exe, dirname=weights_paths, main_program=tp)
    else:
        fluid.io.load_params(exe, dirname="", filename=weights_paths, main_program=tp)


def load_model(weights_paths=' '):
    if weights_paths.endswith('/'):
        # if os.path.exists(weights_paths + '/persistabels_model'):
        #     print('restore model successful')
            fluid.io.load_persistables(exe, dirname=weights_paths, main_program=tp)
            print('restore model successful')
        # else:
        #     pass
    else:
        fluid.io.load_persistables(exe, dirname="", filename=weights_paths, main_program=tp)

def load_model1(exe,checkpoint_dir):
    if os.path.exists(checkpoint_dir ):
        fluid.io.load_params(exe, dirname=checkpoint_dir, main_program=tp)
    else:
        pass
def softmax_cross_entropy(y_true, y_pred, gramm = 2):
    y_pred = fluid.layers.transpose(y_pred, [0, 2, 3, 1])
    y_pred = fluid.layers.reshape(y_pred, [-1, num_classes])
    y_true = fluid.layers.reshape(y_true, [-1, 1])
    y_true = fluid.layers.cast(y_true, 'int64')
    #one_hot =
    # y_true = fluid.layers.reshape(x=y_true, shape=[-1, 9])
    # y_pred = fluid.layers.reshape(x=y_pred, shape=[-1, 9])

    result0 = fluid.layers.softmax_with_cross_entropy(logits=y_pred, label=y_true, soft_label=False)

    no_grad_set.add(y_true.name)
    return fluid.layers.mean(result0)


def focal_loss(y_true, y_pred):
    y_pred = fluid.layers.transpose(y_pred, [0, 2, 3, 1])
    model_out = fluid.layers.elementwise_add(y_pred, fluid.layers.assign(np.array([0.00001], dtype=np.float32)))
    ce_log = fluid.layers.elementwise_mul(fluid.layers.log(y_pred),
                                          fluid.layers.assign(np.array([-1.0], dtype=np.float32)))
    ce = fluid.layers.elementwise_mul(y_true, ce_log)

    tsub = fluid.layers.elementwise_sub(model_out, fluid.layers.assign(np.array([1.0], dtype=np.float32)))
    tsub = fluid.layers.elementwise_mul(tsub, fluid.layers.assign(np.array([-1.0], dtype=np.float32)))
    tpow = fluid.layers.elementwise_pow(tsub, fluid.layers.assign(np.array([2.0], dtype=np.float32)))
    weight = fluid.layers.elementwise_mul(y_true, tpow)
    focal = fluid.layers.elementwise_mul(weight, ce)
    reduced_fl = fluid.layers.reduce_max(focal, -1)
    score = fluid.layers.reduce_mean(reduced_fl, dim=-1)
    return score


def loss(logit, label):
    label_nignore = (label < num_classes).astype('float32')
    label = fluid.layers.elementwise_min(
        label,
        fluid.layers.assign(np.array(
            [num_classes - 1], dtype=np.int32)))
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    loss = fluid.layers.softmax_with_cross_entropy(logit, label)
    loss = loss * label_nignore
    no_grad_set.add(label_nignore.name)
    no_grad_set.add(label.name)
    return loss, label_nignore


models.clean()
models.bn_momentum = 0.9997
models.dropout_keep_prop = 0.9
deeplabv3p = models.deeplabv3p

sp = fluid.Program()
tp = fluid.Program()
base_lr = 0.0001
total_step = 1000
image_shape = [1024, 1024]
lable_shape = [1024, 1024]
num_classes = 9
weight_decay = 0.00004
no_grad_set = set()

parallel_flag = False
use_gpu = True

train_initflag = True
# train_datapath = '/config/baiduLandataset/'
model_save_dir = '/home/zhngqn/zgy_RUIMING/baidu_city/deeplabv3plus_lanedetection_loss_rpc/'
# pretrain_path = 'ction/deeplabv3plus_xception65_initialize.params'
recover_path = '/home/zhngqn/zgy_RUIMING/baidu_city/trainrecovery_softmax/'
weights_path = ' '
if not os.path.exists(recover_path):
                os.makedirs(recover_path)

with fluid.program_guard(tp, sp):
    img = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=lable_shape, dtype='int32')
    logit = deeplabv3p(img)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    # the layers test
    inference_program = fluid.default_main_program().clone(for_test=True)
    print("label " + str(label.shape))
    print("logit " + str(logit.shape))
    print("pred " + str(pred.shape))
    #loss, mask = loss(logit, label)
    loss_mean = softmax_cross_entropy(y_true=label, y_pred=logit)
    lr = fluid.layers.polynomial_decay(base_lr, total_step, end_learning_rate=0, power=0.9)
    # area = fluid.layers.elementwise_max(
    #     fluid.layers.reduce_mean(mask),
    #     fluid.layers.assign(np.array(
    #         [0.1], dtype=np.float32)))
    # loss_mean = fluid.layers.reduce_mean(loss) / area
    opt = fluid.optimizer.Adam(learning_rate=2e-4)
    # opt = fluid.optimizer.Momentum(
    #     lr,
    #     momentum=0.9,
    #     regularization=fluid.regularizer.L2DecayRegularizer(
    #         regularization_coeff=weight_decay), )
    retv = opt.minimize(loss_mean, startup_program=sp, no_grad_set=no_grad_set)
# tp = tp.clone(True)
fluid.memory_optimize(tp, print_log=False, skip_opt_set=[pred.name, loss_mean.name], level=1)

place = fluid.CPUPlace()
if use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

load_model(recover_path)
# load_model1(exe,recover_path)
if parallel_flag:
    exe_p = fluid.ParallelExecutor(use_cuda=True, loss_name=loss_mean.name, main_program=tp)

early_stopcount = 0
last_epoch_acc = 0

filename = '/media/zhngqn/1E3E392E3E390077/baidu/train.txt'
dic_filelist, list_wavmark = get_wav_list(filename)
# lable_data = np.zeros((1, 1024,1024,9), dtype=np.int32)
for pass_id in range(100):
    i = 0
    epoch_acc = 0
    # get train data
    # train_data, train_label_data = data_generator(1024, 1024, 1, dic_filelist, list_wavmark)
    # train_reader = dataprocess.train_generator(train_path=train_datapath,
    #                                            batch_size=1)

    avr_loss = 0
    prev_start_time = time.time()
    for batch_id in range(40000):
        image_data, lable_data = data_generator(1024, 1024,1, dic_filelist, list_wavmark)
        # lable = enc.fit(lable[0,:,:])  # fit来学习编码
        # print('label',enc.transform(lable).toarray().shape)
        # lable_data[0,:,:,:]= enc.transform(lable).toarray()
        # print('image_data',image_data.shape)
        # print('lable_data', lable_data.shape)
        image_data = np.transpose(image_data, (0, 3, 1, 2))
        if parallel_flag:
            retv = exe_p.run(fetch_list=[pred.name, loss_mean.name],
                             feed={'img': image_data,
                                   'label': lable_data})
        else:
            retv = exe.run(tp,
                           feed={'img': image_data,
                                 'label': lable_data},
                           fetch_list=[pred.name, loss_mean.name])

        avr_loss = avr_loss + retv[1]
        # loss = exx.run(
        #                 feed={'input': image_data, 'lable': lable_data},
        #                 fetch_list=[avg_cost.name])
        if batch_id % 1000 == 0:
            avr_loss = avr_loss/1000.0
            epochs_save_path = model_save_dir + "epoch_" + str(pass_id) + '_' + str(batch_id) + "_acc_" + str(int(avr_loss * 1000)) + "/"
            avr_loss = 0
            if not os.path.exists(epochs_save_path):
                os.makedirs(epochs_save_path)
            epochs_save_path = epochs_save_path + "/"
            save_model(epochs_save_path)
        if batch_id % 100 == 0:
            # np.save("/data/modelsave/deeplabv3plus_lanedetection_loss_rpc/result/" + str(pass_id) + '_' + str(i) + 'data.npy',
            #         image_data)
            # np.save("/data/modelsave/deeplabv3plus_lanedetection_loss_rpc/result/" + str(pass_id) + '_' + str(i) + 'lable.npy',
            #         lable_data)
            # np.save("/data/modelsave/deeplabv3plus_lanedetection_loss_rpc/result/" + str(pass_id) + '_' + str(i) + '.npy',
            #         retv[0])
            save_model(recover_path)

        end_time = time.time()
        # epoch_acc += acc
        print("Epochs {:d}===>batch {:d} loss {:.6f}  step_time_cost: {:.3f}".format(pass_id,
                                                                                     batch_id,
                                                                                     np.mean(retv[1]),
                                                                                     end_time - prev_start_time))
        prev_start_time = end_time


