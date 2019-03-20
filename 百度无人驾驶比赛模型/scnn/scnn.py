import random
import cv2
import numpy as np
import paddle
import PIL.Image
import paddle.fluid as fluid
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
def resize_image(img, target_size1, target_size2):
    #     img = img.resize((target_size1, target_size2), PIL.Image.LANCZOS)
    img = cv2.resize(img, (target_size2, target_size1),interpolation=cv2.INTER_NEAREST)
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
    train_data = cv2.imread('/home/zhngqn/zgy_RUIMING/baidu_city'+train_image)
    #     train_data = PIL.Image.open(train_image)
    image = resize_image(train_data, Height, Width)
    # print('image：', image.shape)
    # train_data = fluid.layers.resize_bilinear(train_data, out_shape=[1024,512])
    label_image = cv2.imread('/home/zhngqn/zgy_RUIMING/baidu_city'+label_image)
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
        # X = X[:, :, :, ::-1].astype(np.float32) / (255.0 / 2) - 1
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
        ## 输出的channel = num_filters * 2
        conv2 = self.bottleneck_block(input=conv, num_filters=64, stride=2, trainable=False)  # 42 X
        conv4 = self.bottleneck_block(input=conv2, num_filters=128, stride=2, trainable=False)  # 8 X
        conv8 = self.bottleneck_block(input=conv4, num_filters=128, stride=2, trainable=False)  # 16 X ,
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
            padding=(filter_size - 1) // 2,  # 可能有问题，需要重新设计padding???
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


"""每片之间的卷积"""


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
由上到下，下到上的卷积
"""


def SCNN_D_U(input_data, C_size,H_size, W_size):  # 输入切片的高度
    """
    0,1,2,3 to 0,2,1,3
    """
   # print('input_data',input_data)
    x_transposed = fluid.layers.transpose(input_data, perm=[0, 2, 1, 3])  ## NCHW --> NHCW
   # print('x_transposed', x_transposed)
    axes1 = [1]
    layers_concat = list()
    layers_result_concat = list()

    """SCNN_D"""
    # lenth = range(0,H_size)
    for i in range(0, H_size):
        result = fluid.layers.slice(input=x_transposed, axes=axes1, starts=[i], ends=[i + 1])  # 切片
        # 卷积操作
        if i == 0:  # 第一片就是原始值做卷积
            layers_concat.append(result)  # 保存卷积后的片
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """片于片之间的卷积"""

            scnn_covn2d = SCNN('scnn_covn_D', layers_concat[-1], num_filters=1, filter_size=[3, W_size],
                               padding=[1, 0], stride=1, bias_attr=True, act="relu")
            #print('scnn_covn2d_before:',scnn_covn2d)
            # scnn_covn2d = fluid.layers.reshape(scnn_covn2d,[0,32])#三维变成一个维度
            #print('scnn_covn2d',scnn_covn2d)
            scnn_covn2d = fluid.layers.reshape(scnn_covn2d, [0, C_size],)  # 三维变成一个维度
            #print('scnn_covn2d_after:',scnn_covn2d)
            #             print('result_before:', result)
            result = fluid.layers.transpose(result, perm=[0, 2, 1, 3])
            #             print('result_after:', result)
            # print("result_after:", result)
            # print("scnn_covn2d:", scnn_covn2d)
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # 在最后一个维度上做广播
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            #             print('scnn_covn2d:', scnn_covn2d)
            # scnn_covn2d_last = scnn_covn2d #更新保存上一片
            layers_concat.append(scnn_covn2d)  # 保存卷积后的片
            # layers_concat.append(scnn_covn2d)

    """SCNN_U"""
    for i in range(H_size - 1, -1, -1):
        # print(i)
        result = layers_concat[i]  # 切片
        # 卷积操作
        if i == H_size - 1:  # 第一片就是原始值做卷积
            # print(i)
            layers_result_concat.append(result)  # 保存卷积后的片
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """片于片之间的卷积"""
            #  print(i)
            scnn_covn2d = SCNN('scnn_covn_U', layers_result_concat[-1], num_filters=1, filter_size=[3, W_size],
                               padding=[1, 0], stride=1, bias_attr=True, act="relu")
            # print('scnn_covn2d1',scnn_covn2d)
            scnn_covn2d = fluid.layers.reshape(scnn_covn2d, [0, C_size])  # 三维变成一个维度
            result = fluid.layers.transpose(result, perm=[0, 2, 1, 3])
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # 在最后一个维度上做广播
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            # scnn_covn2d_last = scnn_covn2d #更新保存上一片
            layers_result_concat.append(scnn_covn2d)  # 保存卷积后的片

    return fluid.layers.concat(input=layers_result_concat, axis=1)

    # print('layers_concat11',out)
    # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3], padding=[0,1],stride=1,bias_attr = True,act = "relu")
    # print('scnn_covn2d',scnn_covn2d)


"""
由左到右，右到左的卷积
"""


def SCNN_R_L(input_data, C_size,H_size, W_size):  # 输入切片的高度
    """
    0,1,2,3 to 0,3,2,1    """

    x_transposed = fluid.layers.transpose(input_data, perm=[0, 3, 2, 1])  ## NCHW -->NWCH

    axes1 = [1]
    layers_concat = list()
    layers_result_concat = list()

    """SCNN_R"""
    # lenth = range(0,H_size)
    for i in range(0, W_size):
        result = fluid.layers.slice(input=x_transposed, axes=axes1, starts=[i], ends=[i + 1])  # 切片
        # 卷积操作
        if i == 0:  # 第一片就是原始值做卷积
            layers_concat.append(result)  # 保存卷积后的片
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """片于片之间的卷积"""
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
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # 在最后一个维度上做广播
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            # scnn_covn2d_last = scnn_covn2d #更新保存上一片
            layers_concat.append(scnn_covn2d)  # 保存卷积后的片

    """SCNN_L"""
    for i in range(W_size - 1, -1, -1):
        # print(i)
        result = layers_concat[i]  # 切片
        # 卷积操作
        if i == W_size - 1:  # 第一片就是原始值做卷积
            # print(i)
            layers_result_concat.append(result)  # 保存卷积后的片
            # scnn_covn2d_last = result
            # scnn_covn2d = SCNN_D('scnn_covn2d',result,num_filters =1,filter_size = [32,3],
            #   padding=[0,1],stride=1,bias_attr = True,act = "relu")
        else:
            """片于片之间的卷积"""
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
            scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d, axis=0,act='relu')  # 在最后一个维度上做广播
            #             scnn_covn2d = fluid.layers.elementwise_add(result, scnn_covn2d)
            scnn_covn2d = fluid.layers.transpose(scnn_covn2d, perm=[0, 2, 1, 3])
            #             print("scnn_covn2d:",scnn_covn2d)
            # scnn_covn2d_last = scnn_covn2d #更新保存上一片
            layers_result_concat.append(scnn_covn2d)  # 保存卷积后的片

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
   # net = batch_normalization(net, relu=True, name='conv2d_transpose_bn4')
    net = fluid.layers.conv2d(net, num_classes, 1)
    # net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[512, 1024])
    # net = batch_normalization(net, relu=True, name='conv2d_transpose_bn4')
    # print('net',net)
    #net = fluid.layers.image_resize(net, out_shape=[512, 1024], resample='BILINEAR')
    return net

def save_model(exe,save_dir):
    if os.path.exists(save_dir):
        fluid.io.save_params(exe, dirname=save_dir, main_program=fluid.default_main_program())
    else:
        pass

# global configure

learning_rate = 5e-4
batch_size = 1
num_classes = 9
Image_Height = 512
Image_Width = 1024
num_pass = 100
C_size = 256
checkpoint_dir = '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/SCNN/scnn_checkpoints'
# inference_dir = '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/scnn_inference_model'
save_dir= '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/SCNN/savepath'

# data layer
inputs = fluid.layers.data(name='img', shape=[3, Image_Height, Image_Width], dtype='float32')
# inputs = fluid.layers.image_resize(img, out_shape=[512,256], resample='BILINEAR')  ##   ???
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# infer
predict = output_layers(inputs, C_size, Image_Height // 8, Image_Width // 8,num_classes)
print(predict)

## reshape logits into [batch_zize*H*W, num_classes]
predict = fluid.layers.transpose(predict, [0, 2, 3, 1])  ## NCHW --> NHWC
predict_reshape = fluid.layers.reshape(x=predict, shape=[-1, num_classes])
predict_reshape = fluid.layers.softmax(predict_reshape)
print('predict_reshape:', predict_reshape)

# loss function
print('predict_reshape', predict_reshape)
print('label', label)
cost = fluid.layers.cross_entropy(predict_reshape, label,
                                  soft_label=False)  ## as same as tf.sparse_softmax_cross_entopy_with_logits()
avg_cost = fluid.layers.mean(cost)
print('avg_cost:', avg_cost)

# accuracy
weight_decay = 0.00004
# acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
acc = fluid.layers.accuracy(input=predict_reshape, label=label, k=1)
print('acc:', acc)
# get test program
test_program = fluid.default_main_program().clone(for_test=True)
# optimizer
optimizer = fluid.optimizer.Momentum(
    learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=weight_decay), )
opts = optimizer.minimize(avg_cost)

# get test program
# test_program = fluid.default_main_program().clone(for_test=True)

# run in CPUPlace
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# define a executor
exe = fluid.Executor(place)

# parameters initialize globally
exe.run(fluid.default_startup_program())
# startup_prog = fluid.default_startup_program()
# exe.run(startup_prog)
# fluid.io.load_persistables(exe, checkpoint_dir, startup_prog)
prog = fluid.default_main_program()
# fluid.io.load_persistables(executor=exe, dirname=checkpoint_dir,
#                            main_program=None)
if os.path.exists(checkpoint_dir + '/persistabels_model'):
    fluid.io.load_persistables(exe, checkpoint_dir + '/persistabels_model', main_program=prog)
else:
    pass
# define input data
feeder = fluid.DataFeeder(place=place, feed_list=[inputs, label])
# [inference_program, feed_target_names, fetch_targets]=fluid.io.load_inference_model(dirname=checkpoint_dir+'/', executor=exe)
filename = '/home/zhngqn/zgy_RUIMING/baidu_city/train.txt'
dic_filelist, list_wavmark = get_wav_list(filename)
# begin to train
for pass_id in range(num_pass):
    for batch_id in range(20000):
        start = time.time()
        #         training process
        #num = random.randint(0, 21914)
        #print(dic_filelist[num],list_wavmark[num])
        train_data, train_label_data = data_generator(Image_Height, Image_Width, batch_size, dic_filelist,list_wavmark)
        end = time.time()
        # train_data transpose into NCHW
        train_data = np.transpose(train_data, (0, 3, 1, 2))
        # train_label_data = np.transpose(train_label_data, axes=[0, ])
        train_label_data = np.reshape(train_label_data, (-1, 1))
        # print('train_data, train_label:', [train_data.shape, train_label_data.shape])
        #         train_data = np.random.uniform(0, 1, (batch_size, 3, Image_Height, Image_Width)).astype(np.float32)
        #         train_label_data = np.zeros((5786640 * batch_size, 1)).astype(np.int64).reshape(5786640 * batch_size, 1)

        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed={'img': train_data,
                                              'label': train_label_data},
                                        fetch_list=[avg_cost.name, acc.name])
        # end = time.time()
        # np.size(train_label_data[train_label_data ==0])/np.size(train_label_data)
        base_acc = np.size(train_label_data[train_label_data ==0])/np.size(train_label_data)
        if batch_id % 10 == 0:
            print("Pass: %d, Batch: %d, Train_Cost: %0.5f, Train_Accuracy: %0.5f,time: %0.5f" %
                  (pass_id, batch_id, train_cost[0],(train_acc[0]-base_acc)/(1.0-base_acc) ,end-start))

        # save checkpoints
        #         if pass_id % 10 == 0 and batch_id == 0:
        if batch_id % 50 == 0:
            fluid.io.save_persistables(executor=exe,
                                       dirname=checkpoint_dir + '/persistabels_model',
                                       main_program=fluid.default_main_program()
                                       )
            # fluid.io.save_inference_model(dirname=inference_dir,
            #                               feeded_var_names=['img'],
            #                               target_vars=[predict],
            #                               executor=exe,
            #                               main_program=None,
            #                               model_filename=None,
            #                               params_filename=None,
            #                               export_for_deployment=True)
            save_model(exe, save_dir)