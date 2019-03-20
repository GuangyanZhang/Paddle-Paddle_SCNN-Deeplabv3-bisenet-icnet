"""Trainer for ICNet model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from version0906.icnet1 import icnet
from version0906.deeplabd.data_generater1 import *
import argparse
import functools
import sys
import os
import time
import paddle.fluid as fluid
import numpy as np
from version0906.utils import add_arguments, print_arguments, get_feeder_data
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu

if 'ce_mode' in os.environ:
    np.random.seed(10)
    fluid.default_startup_program().random_seed = 90

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   4,         "Minibatch size.")
add_arg('checkpoint_path',   str,   '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/icenet',       "Checkpoint svae path.")
add_arg('init_model',        str,   '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/icenet/icnet',       "Pretrain model path.")
add_arg('use_gpu',           bool,  True,       "Whether use GPU to train.")
add_arg('random_mirror',     bool,  True,       "Whether prepare by random mirror.")
add_arg('random_scaling',    bool,  True,       "Whether prepare by random scaling.")
# yapf: enable

LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0
LEARNING_RATE = 0.003
POWER = 0.9
LOG_PERIOD = 10
CHECKPOINT_PERIOD = 5000
TOTAL_STEP = 100
# icnet.dropout_keep_prop = 0.5
# icnet.is_train = True

no_grad_set = []


def create_loss(predict, label,num_classes):
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    label = fluid.layers.reshape(label, shape=[-1, 1])
    label = fluid.layers.cast(label, dtype="int64")
    loss = fluid.layers.softmax_with_cross_entropy(predict, label)
    no_grad_set.append(label.name)
    return fluid.layers.reduce_mean(loss)


def poly_decay():
    global_step = _decay_step_counter()
    with init_on_cpu():
        decayed_lr = LEARNING_RATE * (fluid.layers.pow(
            (1 - global_step / TOTAL_STEP), POWER))
    return decayed_lr


def train(args):
    image_shape = [512,1024]
    num_classes = 9
    # define network
    images = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=image_shape, dtype='int32')

    sub4_out, sub24_out, sub124_out = icnet(
        images, num_classes, np.array(image_shape).astype("float32"))
    print('sub4_out',sub24_out)

    loss_sub4 = create_loss(fluid.layers.resize_bilinear(sub4_out, out_shape=image_shape),label, num_classes)
    loss_sub24 = create_loss(fluid.layers.resize_bilinear(sub24_out, out_shape=image_shape), label,num_classes)
    loss_sub124 = create_loss(sub124_out, label,num_classes)
    reduced_loss = LAMBDA1 * loss_sub4 + LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124

    regularizer = fluid.regularizer.L2Decay(0.0001)
    # optimizer = fluid.optimizer.Momentum(
    #     learning_rate=poly_decay(), momentum=0.9, regularization=regularizer)
    optimizer = fluid.optimizer.Adam(0.0001)
    _, params_grads = optimizer.minimize(reduced_loss, no_grad_set=no_grad_set)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    if args.init_model is not None:
        print("load model from: %s" % args.init_model)
        sys.stdout.flush()
        fluid.io.load_params(exe, args.init_model)

    iter_id = 0
    t_loss = 0.
    sub4_loss = 0.
    sub24_loss = 0.
    sub124_loss = 0.
    filename = '/media/zhngqn/1E3E392E3E390077/baidu/train.txt'
    dic_filelist, list_wavmark = get_wav_list(filename)
    prev_start_time = time.time()
    start_time = time.time()
    while True:
        # train a pass
        for iter_id in range(20000):
            # if iter_id > 20000:
            #     end_time = time.time()
            #     print("kpis	train_duration	%f" % (end_time - start_time))
            #     return
            # iter_id += 1
            start = time.time()
            image_data, train_label_data = data_generator(512, 1024, 1, dic_filelist, list_wavmark)
            image_data = np.transpose(image_data, (0, 3, 1, 2))
            # print(iter_id)
            # print('iter_id',image_data.shape)
            results = exe.run(
                program=fluid.default_main_program(),
                feed={'img': image_data,
                      'label': train_label_data},
                fetch_list=[reduced_loss, loss_sub4, loss_sub24, loss_sub124])
            t_loss += results[0]
            sub4_loss += results[1]
            sub24_loss += results[2]
            sub124_loss += results[3]
            # training log
            # print(LOG_PERIOD)
            if iter_id % LOG_PERIOD == 0:
                end = time.time()
                print(
                    "Iter[%d]; train loss: %.3f; sub4_loss: %.3f; sub24_loss: %.3f; sub124_loss: %.3f"
                    % (iter_id, t_loss / LOG_PERIOD, sub4_loss / LOG_PERIOD,
                       sub24_loss / LOG_PERIOD, sub124_loss / LOG_PERIOD))
                print("kpis	train_cost	%f; time: %.3f" % (t_loss / LOG_PERIOD,end -start))

                t_loss = 0.
                sub4_loss = 0.
                sub24_loss = 0.
                sub124_loss = 0.
                sys.stdout.flush()

            if iter_id % CHECKPOINT_PERIOD == 0 and args.checkpoint_path is not None:
                dir_name = args.checkpoint_path + "/" + 'icnet'
                # print(dir_name)
                fluid.io.save_persistables(exe, dirname=dir_name)
                print("Saved checkpoint: %s" % (dir_name))


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == "__main__":
    main()