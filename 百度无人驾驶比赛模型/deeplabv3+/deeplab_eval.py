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
import paddle
import paddle.fluid as fluid
import numpy as np
from version0906.deeplabd import modelsd as models
import time
from version0906.deeplabd.data_generater1 import *
# import dataprocess
# from visualdl import LogWriter
import matplotlib.pyplot as plt

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
            print('restore model successful')
            fluid.io.load_persistables(exe, dirname=weights_paths, main_program=tp)
        # else:
        #     pass
    else:
        fluid.io.load_persistables(exe, dirname="", filename=weights_paths, main_program=tp)


def softmax_cross_entropy(y_true, y_pred, gramm = 2):
    y_pred = fluid.layers.transpose(y_pred, [0, 2, 3, 1])
    y_pred = fluid.layers.reshape(y_pred, [-1, num_classes])
    y_true = fluid.layers.reshape(y_true, [-1, 1])
    y_true = fluid.layers.cast(y_true, 'int64')
    # y_true = fluid.layers.reshape(x=y_true, shape=[-1, 9])
    # y_pred = fluid.layers.reshape(x=y_pred, shape=[-1, 9])

    result0 = fluid.layers.softmax_with_cross_entropy(logits=y_pred, label=y_true, soft_label=False)

    no_grad_set.add(y_true.name)
    return fluid.layers.mean(result0)


def label_mapping(gts):

    colorize = np.zeros([9,3],'uint8')
    colorize[0,:] = [ 0, 0,  0]
    colorize[1,:] = [70, 130,180]
    colorize[2,:] = [  0, 0, 142]
    colorize[3,:] = [  153,  153, 153]
    colorize[4,:] = [ 102, 102,  156]
    colorize[5,:] = [128, 64, 128]
    colorize[6,:] = [ 190, 153, 153]
    colorize[7,:] = [  255, 165, 100]
    colorize[8,:] = [  173,   132, 190]

    ims = colorize[gts,:].reshape([gts.shape[0],gts.shape[1],3])
    return ims

def resize_image(img, target_size1, target_size2):
    #     img = img.resize((target_size1, target_size2), PIL.Image.LANCZOS)
    img = cv2.resize(img, (target_size2, target_size1),interpolation=cv2.INTER_NEAREST)
    return img

def label_result(gts):

    colorize = np.zeros([9,1],'uint8')
    colorize[0,:] = 0
    colorize[1,:] = 200
    colorize[2,:] = 203
    colorize[3,:] = 217
    colorize[4,:] = 218
    colorize[5,:] = 210
    colorize[6,:] = 214
    colorize[7,:] = 220
    colorize[8,:] = 223

    ims = colorize[gts,:].reshape([gts.shape[0],gts.shape[1],1])
    return ims
def label_result1(gts):

    colorize = np.zeros([256,1],'uint8')
    colorize[0,:] =   0
    colorize[200,:] = 200
    colorize[201,:] = 203
    colorize[215,:] = 217
    colorize[218,:] = 218
    colorize[210,:] = 210
    colorize[214,:] = 214
    colorize[202,:] = 220
    colorize[205,:] = 205

    ims = colorize[gts,:].reshape([gts.shape[0],gts.shape[1],1])
    return ims


models.clean()
models.bn_momentum = 0.9997
models.dropout_keep_prop = 1.0
models.is_train = False
deeplabv3p = models.deeplabv3p

sp = fluid.Program()
tp = fluid.Program()
base_lr = 0.0001
total_step = 1000
image_shape = [1024, 1024]
lable_shape = [1024, 1024]
num_classes = 9
weight_decay = 0.00004
batch_size = 1
no_grad_set = set()

parallel_flag = False
use_gpu = True


recover_path = '/home/zhngqn/zgy_RUIMING/baidu_city/trainrecovery_softmax/'


with fluid.program_guard(tp, sp):
    img = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    # label = fluid.layers.data(name='label', shape=lable_shape, dtype='int32')
    logit = deeplabv3p(img)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    # the layers test
    # inference_program = fluid.default_main_program().clone(for_test=True)
    # print("label " + str(label.shape))
    # print("logit " + str(logit.shape))
    # print("pred " + str(pred.shape))
    #loss, mask = loss(logit, label)
    # loss_mean = softmax_cross_entropy(y_true=label, y_pred=logit)
    # lr = fluid.layers.polynomial_decay(base_lr, total_step, end_learning_rate=0, power=0.9)
    # area = fluid.layers.elementwise_max(
    #     fluid.layers.reduce_mean(mask),
    #     fluid.layers.assign(np.array(
    #         [0.1], dtype=np.float32)))
    # loss_mean = fluid.layers.reduce_mean(loss) / area
    # opt = fluid.optimizer.Adam(learning_rate=2e-4)
    # opt = fluid.optimizer.Momentum(
    #     lr,
    #     momentum=0.9,
    #     regularization=fluid.regularizer.L2DecayRegularizer(
    #         regularization_coeff=weight_decay), )
    # retv = opt.minimize(loss_mean, startup_program=sp, no_grad_set=no_grad_set)
# tp = tp.clone(True)

fluid.memory_optimize(tp, print_log=False, skip_opt_set=[pred.name], level=1)

place = fluid.CPUPlace()
if use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

load_model(recover_path)

if parallel_flag:
    exe_p = fluid.ParallelExecutor(use_cuda=True, loss_name=loss_mean.name, main_program=tp)

early_stopcount = 0
last_epoch_acc = 0


if 1:

    filename = '/media/zhngqn/1E3E392E3E390077/baidu/train.txt'
    dic_filelist, list_wavmark = get_wav_list(filename)
    prev_start_time = time.time()
    image_data, train_label_data = data_generator(1024, 1024, 1, dic_filelist,
                                                  list_wavmark)
    train_data = np.array(image_data).reshape(1024, 1024,3)
    train_data = resize_image(train_data, 1710, 3384)

    image_data = np.transpose(image_data, (0, 3, 1, 2))
    print('image_data',image_data.shape)
    # retv = exe.run(tp,
    #                feed={'img': image_data},
    #                fetch_list=[pred.name])
    retv = exe.run(tp,
                   feed={'img': image_data},
                   fetch_list=[pred.name])
    #print(result[0].shape)

    result = np.array(retv).reshape(1024, 1024)
    # train_data_really = np.array(train_data).reshape(1710,3384,3)
    # print(train_label_data.shape)
    train_label_data = np.array(train_label_data).reshape(1024, 1024)
    print(train_label_data.shape)
    # print('im=',im.shape)
    im = label_mapping(result)
    im = resize_image(im, 1710, 3384)
    label =  np.array(label_mapping(train_label_data))
    # print('im=',im.shape)
    label = resize_image(label, 1710, 3384)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.show()

if 0:
    test_path = '/media/zhngqn/1E3E392E3E390077/baidu/ColorImage'
    save_path = '/media/zhngqn/新加卷/baidu/test'
    #def get_result(test_path):
    filelist_iamge = os.listdir(test_path)
    for i in range(len(filelist_iamge)):#len(filelist_iamge)
            start = time.time()
            path = test_path+'/'+filelist_iamge[i]
            test_result = save_path + '/' + filelist_iamge[i]
            test_result = test_result.replace('jpg', 'png')
            # print(path)
            # valid_data = []
            valid_data =cv2.imread(path)
            # test_data = cv2.imread(path)
            valid_data = resize_image(valid_data,1024,1024)
            X = np.zeros((batch_size, 1024, 1024, 3), dtype=np.float32)
            X[0, 0:len(valid_data)] = valid_data
            X = np.transpose(X, (0, 3, 1, 2)).astype('float32')
            # print(i,type(X))

            result = exe.run(tp,
                   feed={'img': X},
                   fetch_list=[pred.name])
            result= np.array(result).reshape(1024,1024)
    # label = np.array(train_label_data).reshape(Image_Height,Image_Width)
    # print(pred[pred>0])
            #im = label_mapping(pred)
            # im = np.array(label_mapping(result)).reshape(1024,1024,3)
            im = np.array(label_result(result)).reshape(1024,1024)
            # print('im=',im.shape)
            im = resize_image(im,1710,3384)
            print(im.shape)
            cv2.imwrite(test_result, im)
            end = time.time()
            print("Pass: %d, time: %0.5f" %(i,  end - start))
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

if 0:
        test_path = '/media/zhngqn/新加卷/baidu/test'
        save_path = '/media/zhngqn/新加卷/baidu/test1'
        # def get_result(test_path):
        filelist_iamge = os.listdir(test_path)
        for i in range(len(filelist_iamge)):  # len(filelist_iamge)
            start = time.time()
            path = test_path + '/' + filelist_iamge[i]
            test_result = save_path + '/' + filelist_iamge[i]
            # print(path)
            valid_data = cv2.imread(path)
            print(valid_data.shape)
            im = np.array(label_result1(valid_data[:,:,0])).reshape(1710, 3384)
            print(im.shape)
            cv2.imwrite(test_result, im)
            end = time.time()
            print("Pass: %d, time: %0.5f" % (i, end - start))
