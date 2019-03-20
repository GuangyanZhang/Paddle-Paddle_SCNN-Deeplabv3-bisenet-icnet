import random
import cv2
import numpy as np
import paddle
import PIL.Image
import paddle.fluid as fluid
import time
import os
np.set_printoptions(threshold=np.nan)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
def resize_image(img, target_size1, target_size2):
    #     img = img.resize((target_size1, target_size2), PIL.Image.LANCZOS)
    img = cv2.resize(img, (target_size2, target_size1))
    return img

def batch_normalization(x, relu = False, name = ''):
    if relu:
        return fluid.layers.batch_norm(x, act = 'relu', name = name)
    else:
        return fluid.layers.batch_norm(x, name = name)

def get_wav_list(filename):
      '''
      读取一个wav文件列表，返回一个存储该列表的字典类型值
      ps:在数据中专门有几个文件用于存放用于训练、验证和测试的wav文件列表
      '''

      txt_obj = open(filename, 'r')  # 打开文件并读入
      txt_text = txt_obj.read()
      txt_lines = txt_text.split('\n')  # 文本分割

      dic_filelist = []  # 初始化字典
      list_wavmark = []  # 初始化wav列表
      j = 0
      for i in txt_lines:
          if (i != ''):
              txt_l = i.split(' ')

              dic_filelist.append(txt_l[0])
              list_wavmark.append(txt_l[1])
              #j = j + 1
              #print(list_wavmark[str(j)])

      txt_obj.close()
      return dic_filelist, list_wavmark


def get_data(Height, Width,train_image,label_image):
    # train_image = dic_filelist[num]
    # train_image = list_wavmark[num]
    train_data = cv2.imread('/home/zhngqn/zgy_RUIMING/baidu_city/'+train_image)
    #     train_data = PIL.Image.open(train_image)
    image = resize_image(train_data, Height, Width)
    # print('image：', image.shape)
    # train_data = fluid.layers.resize_bilinear(train_data, out_shape=[1024,512])
    label_image = cv2.imread('/home/zhngqn/zgy_RUIMING/baidu_city/'+label_image)
    # print('label_image_before：', label_image.shape)
    label_image = resize_image(label_image, Height, Width)
    # print('label_image_after：', label_image.shape)
    #     label_image = PIL.Image.open(label_image)
    return image, label_image[:, :, 0]  # 取了一个通道


"""
批量读取图片
"""
def get_random():
    num = random.randint(1, 21913)

    if (num in dic_filelist.keys()):
        return num
    else:
        return [-1], [-1]

def data_generator(Height, Width, batch_size,dic_filelist,list_wavmark):
    X = np.zeros((batch_size, Height, Width, 3), dtype=np.float32)  # batch_size train data
    Y = np.zeros((batch_size, Height, Width), dtype=np.int64)  # batch_size label data
    for i in range(batch_size):
        num = random.randint(1, 21913)
        if num in dic_filelist:
            pass
        else:

            num = random.randint(1, 21913)

        #ran_num = random.randint(0, 3999) #生成一个随机数
        train_data, label_image = get_data(Height, Width,dic_filelist[num],list_wavmark[num])
        # print('train_data：', train_data.shape)
        # print('label_image：', label_image.shape)
        X[i, 0:len(train_data)] = train_data
        Y[i, 0:len(label_image)] = label_image
        X = X[:, :, :, ::-1].astype(np.float32) / (255.0 / 2) - 1
    return X, Y

"""
conv2d + BN(TRUE OR FALSE) + RELU+pool
"""
def conv_layers(layers_name,data,num_filters,num_filter_size = 3, stride=2,pool_stride=2, padding=1,bias_attr = True,act = "relu",Use_BN = True):
    conv2d = fluid.layers.conv2d(input = data,
                                 num_filters = num_filters,
                                 filter_size = num_filter_size,
                                 stride=stride,
                                 padding=padding,
                                 bias_attr = bias_attr,
                                 act = None,
                                 name = layers_name + '_conv2d' )
    if Use_BN:
        BN = fluid.layers.batch_norm(input = conv2d,name = layers_name + '_BN')
    else:
        BN = conv2d
    out_put = fluid.layers.relu(BN,name = layers_name + '_relu')
    return fluid.layers.pool2d(
                  input=out_put,
                  pool_size=2,
                  pool_type='max',
                  pool_stride=pool_stride,
                  global_pooling=False)
    #return out_put


class ResNet():
    """
    2X
    4X
    8X
    16X
    32X
    output16 = 16X
    output32 = 32X
    layers = 22 (2+4*5)
    """

    def __init__(self, is_test=False):
        self.is_test = is_test

    def net(self, input):
        # if layers == 22:
        #     depth = [1, 1, 1, 1]

        conv = self.conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=1, act='relu', trainable=False)
        conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=1, pool_padding=1, pool_type='max')

        # conv2 = self.bottleneck_block(input=conv, num_filters=64, stride=2, trainable=False)#2 X
        ## ???channel = num_filters * 2
        conv2 = self.bottleneck_block(input=conv, num_filters=64, stride=2, trainable=False)  # 42 X
        conv4 = self.bottleneck_block(input=conv2, num_filters=128, stride=2, trainable=False)  # 8 X
        conv8 = self.bottleneck_block(input=conv4, num_filters=256, stride=2, trainable=False)  # 16 X ,
        return conv8

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      trainable=True):
        param_attr = fluid.ParamAttr(trainable=trainable)

        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,  # ????????????padding???
            groups=groups,
            act=None,
            bias_attr=True,
            param_attr=param_attr)
        return fluid.layers.batch_norm(input=conv, act=act, is_test=self.is_test, param_attr=param_attr)

    def shortcut(self, input, ch_out, stride, trainable=True):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride=stride, trainable=trainable)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, trainable=True):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            act='relu',
            trainable=trainable)

        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            trainable=trainable)

        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            stride=1,
            act=None,
            trainable=trainable)
        short = self.shortcut(input, num_filters * 2, stride, trainable=trainable)
        #         print('element_2:', [short, conv2])

        return fluid.layers.elementwise_add(x=conv2, y=short, act='relu')


"""???????"""


def SCNN(layers_name, data, num_filters, filter_size, padding, stride=1, bias_attr=True, act="relu"):
    return fluid.layers.conv2d(input=data,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               bias_attr=bias_attr,
                               act=act,
                               name=layers_name + '_conv2d')


# out_put = fluid.layers.relu(conv2d,name = layers_name + '_relu')

"""
???????????
"""


def SCNN_D_U(input_data, C_size,H_size, W_size):  # ???????
    """
    0,1,2,3 to 0,2,1,3
    """
    print('W_size:', W_size)
    print('input_data:',input_data)
    x_transposed = fluid.layers.transpose(input_data, perm=[0, 2, 1, 3])  ## NCHW --> NHCW
    print('x_transposed', x_transposed)
    axes1 = [1]
    layers_concat = list()
    layers_result_concat = list()

    """SCNN_D"""
    # lenth = range(0,H_size)
    for i in range(0, H_size):
        result = fluid.layers.slice(input=x_transposed, axes=axes1, starts=[i], ends=[i + 1])  # ??
        # ????
        if i == 0:  # ???????????
            layers_concat.append(result)  # ???????
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """????????"""
            print('layers_concat[-1]:', layers_concat[-1])
            scnn_covn2d = SCNN('scnn_covn_D', layers_concat[-1], num_filters=1, filter_size=[3, W_size],
                               padding=[1, 0], stride=1, bias_attr=True, act="relu")
            #print('scnn_covn2d_before:',scnn_covn2d)
            # scnn_covn2d = fluid.layers.reshape(scnn_covn2d,[0,32])#????????
            print('scnn_covn2d',scnn_covn2d)
            scnn_covn2d = fluid.layers.reshape(scnn_covn2d, [0, C_size])  # ????????
            #print('scnn_covn2d_after:',scnn_covn2d)
            #             print('result_before:', result)
            result = fluid.layers.transpose(result, perm=[0, 2, 1, 3])
            #             print('result_after:', result)
            # print("result_after:", result)
            # print("scnn_covn2d:", scnn_covn2d)
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # ???????????
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            #             print('scnn_covn2d:', scnn_covn2d)
            # scnn_covn2d_last = scnn_covn2d #???????
            layers_concat.append(scnn_covn2d)  # ???????
            # layers_concat.append(scnn_covn2d)

    """SCNN_U"""
    for i in range(H_size - 1, -1, -1):
        # print(i)
        result = layers_concat[i]  # ??
        # ????
        if i == H_size - 1:  # ???????????
            # print(i)
            layers_result_concat.append(result)  # ???????
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """????????"""
            #  print(i)
            scnn_covn2d = SCNN('scnn_covn_U', layers_result_concat[-1], num_filters=1, filter_size=[3, W_size],
                               padding=[1, 0], stride=1, bias_attr=True, act="relu")
            # print('scnn_covn2d1',scnn_covn2d)
            scnn_covn2d = fluid.layers.reshape(scnn_covn2d, [0, C_size])  # ????????
            result = fluid.layers.transpose(result, perm=[0, 2, 1, 3])
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # ???????????
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            # scnn_covn2d_last = scnn_covn2d #???????
            layers_result_concat.append(scnn_covn2d)  # ???????

    return fluid.layers.concat(input=layers_result_concat, axis=1)

    # print('layers_concat11',out)
    # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3], padding=[0,1],stride=1,bias_attr = True,act = "relu")
    # print('scnn_covn2d',scnn_covn2d)


"""
???????????
"""


def SCNN_R_L(input_data, C_size,H_size, W_size):  # ???????
    """
    0,1,2,3 to 0,3,2,1    """

    x_transposed = fluid.layers.transpose(input_data, perm=[0, 3, 2, 1])  ## NCHW -->NWCH

    axes1 = [1]
    layers_concat = list()
    layers_result_concat = list()

    """SCNN_R"""
    # lenth = range(0,H_size)
    for i in range(0, W_size):
        result = fluid.layers.slice(input=x_transposed, axes=axes1, starts=[i], ends=[i + 1])  # ??
        # ????
        if i == 0:  # ???????????
            layers_concat.append(result)  # ???????
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """????????"""
            #             if i == 2:
            #                 print('SCNN_R:', layers_concat[-1])
            #             else:
            #                 pass
            scnn_covn2d = SCNN('scnn_covn_R', layers_concat[-1], num_filters=1, filter_size=[3, H_size],
                               padding=[1, 0], stride=1, bias_attr=True, act="relu")

            scnn_covn2d = fluid.layers.reshape(scnn_covn2d, [0, C_size])
            result = fluid.layers.transpose(result, perm=[0, 2, 1, 3])
            # print("result_after:", result)
            # print("scnn_covn2d:", scnn_covn2d)
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # ???????????
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            # scnn_covn2d_last = scnn_covn2d #???????
            layers_concat.append(scnn_covn2d)  # ???????

    """SCNN_L"""
    for i in range(W_size - 1, -1, -1):
        # print(i)
        result = layers_concat[i]  # ??
        # ????
        if i == W_size - 1:  # ???????????
            # print(i)
            layers_result_concat.append(result)  # ???????
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """????????"""
            #  print(i)
            scnn_covn2d = SCNN('scnn_covn_L', layers_result_concat[-1], num_filters=1, filter_size=[3, H_size],
                               padding=[1, 0], stride=1, bias_attr=True, act="relu")
            #             print("scnn_covn2d_before:",scnn_covn2d)
            scnn_covn2d = fluid.layers.reshape(scnn_covn2d, [0, C_size])
            #             print("scnn_covn2d_after:",scnn_covn2d)
            #             print("result_before:",result)
            result = fluid.layers.transpose(result, perm=[0, 2, 1, 3])
            # print("result_after:", result)
            # print("scnn_covn2d:", scnn_covn2d)
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # ???????????
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            #             print("scnn_covn2d:",scnn_covn2d)
            # scnn_covn2d_last = scnn_covn2d #???????
            layers_result_concat.append(scnn_covn2d)  # ???????

    return fluid.layers.concat(input=layers_result_concat, axis=1)


def SCNN_D_U_R_L(input_data,C_size,H_size ,W_size):
    #input_data =  conv_layers( 'layers1',input_image,num_filters = 32, stride=2, padding=1,bias_attr = True)
    #out_data_D_U = SCNN_D_U(input_data,H_size)
    #print('out_data_D_U',out_data_D_U)
    return fluid.layers.transpose(SCNN_R_L(SCNN_D_U(input_data,C_size,H_size,W_size),C_size,H_size,W_size), perm=[0, 2, 3,1])
    #print(out_transposed)


def output_layers(input_data,C_size, H_size,W_size,num_classes):
    model = ResNet(is_test=False)
    # spatial_net = model.bottleneck_block1(inputs)
    end_points_8 = model.net(input_data)
    output_dat = SCNN_D_U_R_L(end_points_8,C_size, H_size,W_size )
    net = batch_normalization(output_dat, relu=True, name='conv2d_transpose_bn1')
    net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[128, 256])
    net = batch_normalization(net, relu=True, name='conv2d_transpose_bn2')
    net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[256, 512])
    net = batch_normalization(net, relu=True, name='conv2d_transpose_bn3')
    net = fluid.layers.image_resize(net, out_shape=[512, 1024], resample='BILINEAR')
    net = batch_normalization(net, relu=True, name='conv2d_transpose_bn4')
    net = fluid.layers.conv2d(net, num_classes, 1,act = 'relu')
    # net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[512, 1024])
    # net = batch_normalization(net, relu=True, name='conv2d_transpose_bn4')
    # print('net',net)
    #net = fluid.layers.image_resize(net, out_shape=[512, 1024], resample='BILINEAR')
    return net
def load_model(exe,checkpoint_dir):
    if os.path.exists(checkpoint_dir ):
        fluid.io.load_params(exe, dirname=checkpoint_dir, main_program=fluid.default_main_program())
    else:
        pass

def save_model(exe,save_dir):
    if os.path.exists(save_dir):
        fluid.io.save_params(exe, dirname=save_dir, main_program=fluid.default_main_program())
    else:
        pass

def label_mapping(gts):

    colorize = np.zeros([9,3],'uint8')
    colorize[0,:] = [ 0, 0,  0]
    colorize[1,:] = [70, 130,180]
    colorize[2,:] = [  0, 0, 142]
    colorize[3,:] = [  153,  153, 153]
    colorize[4,:] = [ 102, 102,  156]
    colorize[5,:] = [128, 64, 128]
    colorize[6,:] = [ 190, 153, 153]
    colorize[7,:] = [  150, 100, 100]
    colorize[8,:] = [  102,   0, 204]

    ims = colorize[gts,:].reshape([gts.shape[0],gts.shape[1],3])
    return ims

def resize_image(img, target_size1, target_size2):
    #     img = img.resize((target_size1, target_size2), PIL.Image.LANCZOS)
    img = cv2.resize(img, (target_size2, target_size1))
    return img

def label_result(gts):

    colorize = np.zeros([9,1],'uint8')
    colorize[0,:] = 0
    colorize[1,:] = 200
    colorize[2,:] = 201
    colorize[3,:] = 215
    colorize[4,:] = 218
    colorize[5,:] = 210
    colorize[6,:] = 214
    colorize[7,:] = 202
    colorize[8,:] = 205

    ims = colorize[gts,:].reshape([gts.shape[0],gts.shape[1],1])
    return ims

# global configure

learning_rate = 1e-4
batch_size = 1
num_classes = 9
Image_Height = 512
Image_Width = 1024
num_pass = 100
C_size = 512

save_dir= '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/savepath'
checkpoint_dir = '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/scnn_checkpoints/persistabels_model'
# data layer
inputs = fluid.layers.data(name='img', shape=[3, Image_Height, Image_Width], dtype='float32')
# inputs = fluid.layers.image_resize(img, out_shape=[512,256], resample='BILINEAR')  ##   ???
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# infer
predict = output_layers(inputs, C_size, Image_Height // 8, Image_Width // 8, num_classes)
# print(predict)

## reshape logits into [batch_zize*H*W, num_classes]
# predict = fluid.layers.argmax(predict, axis=1).astype('int32')
pred = fluid.layers.transpose(predict, [0, 2, 3, 1])  ## NCHW --> NHWC
pred = fluid.layers.argmax(pred, axis=3).astype('int32')
# print('predict=', pred)
# pred = fluid.layers.argmax(predict, axis=3).astype('int32')
# predict_reshape = fluid.layers.reshape(x=predict, shape=[-1, num_classes])
# predict_reshape = fluid.layers.softmax(predict_reshape)

# print('predict_reshape:', predict_reshape)

# loss function
# print('predict_reshape', predict_reshape)
# print('label', label)
# cost = fluid.layers.cross_entropy(predict_reshape, label,
#                                   soft_label=False)  ## as same as tf.sparse_softmax_cross_entopy_with_logits()
# avg_cost = fluid.layers.mean(cost)
# # print('avg_cost:', avg_cost)
#
# # accuracy
# # acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
# acc = fluid.layers.accuracy(input=predict_reshape, label=label, k=1)
# # print('acc:', acc)
# # get test program
# test_program = fluid.default_main_program().clone(for_test=True)
# # optimizer
# optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
# opts = optimizer.minimize(avg_cost)

# get test program
# test_program = fluid.default_main_program().clone(for_test=True)

# run in CPUPlace
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# define a executor
exe = fluid.Executor(place)

# parameters initialize globally
exe.run(fluid.default_startup_program())
startup_prog = fluid.default_startup_program()
# exe.run(startup_prog)
# fluid.io.load_persistables(exe, checkpoint_dir, startup_prog)
# prog = fluid.default_main_program()
# fluid.io.load_persistables(executor=exe, dirname=checkpoint_dir,
#                            main_program=None)

# define input data
load_model(exe,save_dir)

if 1:

    filename = '/home/zhngqn/zgy_RUIMING/baidu_city/train.txt'
    dic_filelist, list_wavmark = get_wav_list(filename)
    # begin to train
    all_correct = np.array([0], dtype=np.int64)
    all_wrong = np.array([0], dtype=np.int64)
    prev_start_time = time.time()
    train_data, train_label_data = data_generator(Image_Height, Image_Width, batch_size, dic_filelist,list_wavmark)
    # end_time = time.time()
    # train_data transpose into NCHW
    train_data = np.transpose(train_data, (0, 3, 1, 2))

    result = exe.run(program=fluid.default_main_program(),
                   feed={'img': train_data},
                   fetch_list=[pred])
    print(np.unique(result[0]))
    result = np.array(result).reshape(Image_Height, Image_Width)
    # print(train_label_data.shape)
    train_label_data = np.array(train_label_data).reshape(Image_Height, Image_Width)
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
    test_path = '/home/zhngqn/zgy_RUIMING/baidu_city/ColorImage'
    save_path = '/home/zhngqn/zgy_RUIMING/baidu_city/test'
    #def get_result(test_path):
    filelist_iamge = os.listdir(test_path)
    for i in range(len(filelist_iamge)):
            start = time.time()
            path = test_path+'/'+filelist_iamge[i]
            test_result = save_path + '/' + filelist_iamge[i]
            test_result = test_result.replace('jpg', 'png')
            # print(path)
            # valid_data = []
            valid_data =cv2.imread(path)
            # test_data = cv2.imread(path)
            valid_data = resize_image(valid_data,Image_Height,Image_Width)
            X = np.zeros((batch_size, Image_Height, Image_Width, 3), dtype=np.float32)
            X[0, 0:len(valid_data)] = valid_data
            X = np.transpose(X, (0, 3, 1, 2)).astype('float32')
            # print(i,type(X))

            result = exe.run(program=fluid.default_main_program(),feed={'img': X},fetch_list=[pred])
            result= np.array(result).reshape(Image_Height,Image_Width)
    # label = np.array(train_label_data).reshape(Image_Height,Image_Width)
    # print(pred[pred>0])
            #im = label_mapping(pred)
            im = np.array(label_result(result)).reshape(Image_Height,Image_Width)
            # print('im=',im.shape)
            im = resize_image(im,1710,3384)
            print(im.shape)
            cv2.imwrite(test_result, im)
            end = time.time()
            print("Pass: %d, time: %0.5f" %(i,  end - start))

