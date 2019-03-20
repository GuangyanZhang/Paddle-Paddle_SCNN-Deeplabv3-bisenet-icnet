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
import matplotlib.pyplot as plt
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
CHECKPOINT_PERIOD = 1000
TOTAL_STEP = 100
# icnet.is_train = False

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

    sub4_out, sub24_out, sub124_out = icnet(
        images, num_classes, np.array(image_shape).astype("float32"))
    pred = fluid.layers.argmax(sub124_out, axis=1).astype('int32')


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
    if 0:

        filename = '/media/zhngqn/1E3E392E3E390077/baidu/train.txt'
        dic_filelist, list_wavmark = get_wav_list(filename)
        image_data, train_label_data = data_generator(512, 1024, 1, dic_filelist, list_wavmark)
        image_data = np.transpose(image_data, (0, 3, 1, 2))
        # print(iter_id)
        # print('iter_id',image_data.shape)
        retv = exe.run(
            program=fluid.default_main_program(),
            feed={'img': image_data,},
            fetch_list=[pred])
        print(np.array(retv).shape)
        result = np.array(retv).reshape(512, 1024)
        # train_data_really = np.array(train_data).reshape(1710,3384,3)
        # print(train_label_data.shape)
        train_label_data = np.array(train_label_data).reshape(512, 1024)
        print(train_label_data.shape)
        # print('im=',im.shape)
        im = label_mapping(result)
        im = resize_image(im, 1710, 3384)
        label = np.array(label_mapping(train_label_data))
        # print('im=',im.shape)
        label = resize_image(label, 1710, 3384)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(label)
        plt.subplot(2, 1, 2)
        plt.imshow(im)
        plt.show()
    if 1:
        test_path = '/media/zhngqn/1E3E392E3E390077/baidu/ColorImage'
        save_path = '/media/zhngqn/新加卷/baidu/test'
        # def get_result(test_path):
        filelist_iamge = os.listdir(test_path)
        for i in range(len(filelist_iamge)):  # len(filelist_iamge)
            start = time.time()
            path = test_path + '/' + filelist_iamge[i]
            test_result = save_path + '/' + filelist_iamge[i]
            test_result = test_result.replace('jpg', 'png')
            # print(path)
            # valid_data = []
            valid_data = cv2.imread(path)
            # test_data = cv2.imread(path)
            valid_data = resize_image(valid_data, 512, 1024)
            X = np.zeros((1, 512, 1024, 3), dtype=np.float32)
            X[0, 0:len(valid_data)] = valid_data
            X = np.transpose(X, (0, 3, 1, 2)).astype('float32')
            # print(i,type(X))

            retv = exe.run(program=fluid.default_main_program(),
                             feed={'img': X},
                             fetch_list=[pred])
            result = np.array(retv).reshape(512, 1024)
            # label = np.array(train_label_data).reshape(Image_Height,Image_Width)
            # print(pred[pred>0])
            # im = label_mapping(pred)
            # im = np.array(label_mapping(result)).reshape(512,1024,3)
            im = np.array(label_result(result)).reshape(512, 1024)
            # print('im=',im.shape)
            im = resize_image(im, 1710, 3384)
            print(im.shape)
            cv2.imwrite(test_result, im)
            end = time.time()
            print("Pass: %d, time: %0.5f" % (i, end - start))
            # im_label = resize_image(im_label,1710,3384)
        # plt.imshow(im)  # 绘图
        # plt.show()
        #
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(test_data-im)
            # plt.subplot(1, 2, 2)
            # plt.imshow(im)
            # plt.show()


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == "__main__":
    main()