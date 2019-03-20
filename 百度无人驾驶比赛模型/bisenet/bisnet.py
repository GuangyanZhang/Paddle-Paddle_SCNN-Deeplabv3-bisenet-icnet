import cv2
import numpy as np
import paddle.fluid as fluid
import os
import random
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
    return X, Y


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
        conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        # conv2 = self.bottleneck_block(input=conv, num_filters=64, stride=2, trainable=False)#2 X
        ## 输出的channel = num_filters * 2
        conv4 = self.bottleneck_block(input=conv, num_filters=64, stride=2, trainable=False)  # 42 X
        conv8 = self.bottleneck_block(input=conv4, num_filters=128, stride=2, trainable=False)  # 8 X
        conv16 = self.bottleneck_block(input=conv8, num_filters=256, stride=2, trainable=False)  # 16 X ,
        conv32 = self.bottleneck_block(input=conv16, num_filters=512, stride=2, trainable=False)  # 32X
        return conv16, conv32

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


def Upsample(inputs, scale):
    """
    对输入的tensor做上采样。
    输入：
        inputs(tensor)
        scale: 放缩的倍数
    输出：
        return（tensor）: 返回上采样的结果
    """
    return fluid.layers.resize_bilinear(input=inputs, scale=scale)


def ConvBlock(inputs, num_filters, kernel_size=3, stride=2):
    """
    基本的卷积模块，操作顺序为conv，batch_norm, relu
    输入：
        inputs(tensor): Input value, a tensor with NCHW format.
        num_filters(integer): the number of filters, which is the same with the output channel.
        kernel_size(int|tuple): the filter size, if kernel_size is a tuple, it must contain two integers, (kernel_size_H, kernel_size_W).
        stride(integer): The stride size. If stride is a tuple, it must contain two integers, (stride_H, stride_W).
                         Otherwise, the stride_H = stride_W = stride. Default: stride = 2.
    输出：
        返回卷积，bn和relu后的结果。
    """
    net = fluid.layers.conv2d(input=inputs, num_filters=num_filters,
                              filter_size=kernel_size, stride=stride, padding=1)
    net = fluid.layers.relu(fluid.layers.batch_norm(net), name='net')
    return net


def AttentionRefinementModule(inputs, num_filters):
    """
    Attention模块
    输入：
        inputs(tensor): input value, a tensor with NCHW format
        num_filters(int): the number of filters in 1x1 convolution
    输出：
        attention 的结果
    """
    # global average pool
    net = fluid.layers.reduce_mean(input=inputs, dim=[2, 3], keep_dim=True)

    # 1x1 conv
    net = fluid.layers.conv2d(input=net, num_filters=num_filters, filter_size=1)
    # batch_norm
    net = fluid.layers.batch_norm(input=net)
    # sigmoid activate
    net = fluid.layers.sigmoid(net)
    # input multiply sigmoid（net)
    #     print('element_3:', [inputs, net])
    net = fluid.layers.elementwise_mul(inputs, net, axis=0)
    return net


# def FeatureFusionModule(input_1, input_2, num_filters):
def FeatureFusionModule(inputs, num_filters):
    #     # concate layer
    #     inputs=fluid.layers.concat([input_1, input_2], axis=3)

    # conv_bn_relu layer
    inputs = ConvBlock(inputs, num_filters=num_filters, kernel_size=[3, 3], stride=1)
    # print('inputs0000000000000000000000')

    # FFM branch
    ## global average pool
    net = fluid.layers.reduce_mean(input=inputs, dim=[2, 3], keep_dim=True)
    ## 1x1 conv
    net = fluid.layers.conv2d(input=net, num_filters=num_filters, filter_size=1)

    ## relu
    net = fluid.layers.relu(net)

    ## 1x1 conv
    net = fluid.layers.conv2d(input=net, num_filters=num_filters, filter_size=1)

    ## sigmoid activate
    net = fluid.layers.sigmoid(net)

    # input multiply sigmoid（net)
    #     print('element_4:', [inputs, net])
    net = fluid.layers.elementwise_mul(inputs, net, axis=0)

    # input add net
    #     print('element_5:', [inputs, net])
    net = fluid.layers.elementwise_add(inputs, net)

    return net


def build_bisenet(inputs, num_classes):
    """
    Builds the BiSeNet model.
    Arguments:
      inputs: The input tensor=
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction
      num_classes: Number of classes
    Returns:
      BiSeNet model
    """

    ### The spatial path
    ### The number of feature maps for each convolution is not specified in the paper
    ### It was chosen here to be equal to the number of feature maps of a classification
    ### model at each corresponding stage
    #     spatial_net = fluid.layers.resize_bilinear(inputs, [Image_Height/8, Image_Width/8])
    #     print('spatial_net_1',spatial_net)

    ## spatial path
    spatial_net = ConvBlock(inputs, num_filters=64, kernel_size=3, stride=2)
    spatial_net = ConvBlock(spatial_net, num_filters=128, kernel_size=3, stride=2)
    spatial_net = ConvBlock(spatial_net, num_filters=256, kernel_size=3, stride=2)
    # print("spatial_net:", spatial_net)

    # spatial_net = fluid.layers.resize_bilinear(spatial_net, [Image_Height/8, Image_Width/8])
    #     print('spatial_net_2',spatial_net)
    ### Context path
    model = ResNet(is_test=False)
    # spatial_net = model.bottleneck_block1(inputs)
    end_points_16, end_points_32 = model.net(inputs)
    net_4 = AttentionRefinementModule(end_points_16, num_filters=512)
    net_5 = AttentionRefinementModule(end_points_32, num_filters=1024)
    global_channels = fluid.layers.reduce_mean(net_5, [2, 3], keep_dim=True)
    net_5_scaled = fluid.layers.elementwise_mul(net_5, global_channels, axis=0)

    ### Combining the paths
    net_4 = Upsample(net_4, scale=2)
    net_5_scaled = Upsample(net_5_scaled, scale=4)
    # print('net_4, net_5:', [net_4, net_5_scaled])
    #  layers_concat = list()
    #  layers_concat.append(spatial_net)
    ## layers_concat.append(net_4)
    # layers_concat.append(net_5_scaled)
    context_net = fluid.layers.concat([spatial_net, net_4, net_5_scaled], axis=1)  #
    # context_net = fluid.layers.concat(input=layers_concat,axis=1)
    # print('context_net', context_net)
    #     context_net = fluid.layers.concat([net_4, net_5_scaled], axis=1)
    #     print('context_net', context_net)
    #     context_net = fluid.layers.concat([spatial_net,context_net], axis=1)
    #     print('context_net2',context_net)

    ### FFM
    #     net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, num_filters=num_classes)
    net = FeatureFusionModule(inputs=context_net, num_filters=num_classes)

    # print('net', net)

    ## [batch_zize, num_filters, 128, 64]

    ### Final upscaling and finish
    #     net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[256, 128])
    # print('conv2d_transpose', net)
    net = batch_normalization(net, relu=True, name='conv2d_transpose_bn1')
    net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[128, 256])
    net = batch_normalization(net, relu=True, name='conv2d_transpose_bn2')
    net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[256, 512])
    net = batch_normalization(net, relu=True, name='conv2d_transpose_bn3')
    #net = fluid.layers.conv2d_transpose(input=net, num_filters=num_classes, output_size=[512, 1024])
    #net = batch_normalization(net, relu=True, name='conv2d_transpose_bn4')
    # print('net',net)
    net = fluid.layers.image_resize(net, out_shape=[512, 1024], resample='BILINEAR')

    net = fluid.layers.conv2d(net, num_classes, 1)
    return net


# global configure
learning_rate = 1e-4
batch_size = 3
num_classes = 9
Image_Height = 512
Image_Width = 1024
num_pass = 100
checkpoint_dir = '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/checkpoints1'
inference_dir = '/home/zhngqn/zgy_RUIMING/code/ASR_v0.6_8k/version0906/inference_model1'

# data layer
inputs = fluid.layers.data(name='img', shape=[3, Image_Height, Image_Width], dtype='float32')
#inputs = fluid.layers.image_resize(img, out_shape=[512,256], resample='BILINEAR')  ##   ???
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# infer
predict = build_bisenet(inputs=inputs, num_classes=num_classes)

## reshape logits into [batch_zize*H*W, num_classes]
# predict = fluid.layers.transpose(predict, [0, 3, 1, 2]) ## NCHW --> NHWC , n*1710*3384*9
# print()
predict = fluid.layers.transpose(predict, [0, 2, 3, 1])  ## NCHW --> NHWC , n*1710*3384*9
# print("predict:", predict)
predict_reshape = fluid.layers.reshape(x=predict, shape=[-1, num_classes])
predict_reshape = fluid.layers.softmax(predict_reshape)
# print('predict_reshape:', predict_reshape)

# loss function
cost = fluid.layers.cross_entropy(predict_reshape, label,soft_label=False)
## as same as tf.sparse_softmax_cross_entopy_with_logits()
avg_cost = fluid.layers.mean(cost)
# print('avg_cost:', avg_cost)

# accuracy
acc = fluid.layers.accuracy(input=predict_reshape, label=label, k=1)
# print('acc:', acc)
test_program = fluid.default_main_program().clone(for_test=True)
# optimizer
optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
opts = optimizer.minimize(avg_cost)

# get test program
# test_program = fluid.default_main_program().clone(for_test=True)

# run in CPUPlace
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# define a executor
exe = fluid.Executor(place)

# parameters initialize globally
# exe.run(fluid.default_startup_program())
startup_prog = fluid.default_startup_program()
prog = fluid.default_main_program()
exe.run(startup_prog)
if os.path.exists(checkpoint_dir + '/persistabels_model'):
    fluid.io.load_persistables(exe, checkpoint_dir + '/persistabels_model', main_program=prog)
else:
    pass
# define input data
# feeder = fluid.DataFeeder(place=place, feed_list=[inputs, label])
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
        if batch_id % 1 == 0:
            print("Pass: %d, Batch: %d, Train_Cost: %0.5f, Train_Accuracy: %0.5f,time: %0.5f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0],end-start))

        # save checkpoints
        #         if pass_id % 10 == 0 and batch_id == 0:
        if batch_id % 10 == 0:
            fluid.io.save_persistables(executor=exe,
                                       dirname=checkpoint_dir + '/persistabels_model',
                                       main_program=fluid.default_main_program()
                                       )
            fluid.io.save_inference_model(dirname=inference_dir,
                                          feeded_var_names=['img'],
                                          target_vars=[predict],
                                          executor=exe,
                                          main_program=None,
                                          model_filename=None,
                                          params_filename=None,
                                          export_for_deployment=True)
        #
        # # test process
        # if batch_id % 25 == 0:
        #     test_data, test_label_data = data_generator(Image_Height, Image_Width, batch_size, is_train=False)
        #
        #     ## train_data transpose into NCHW
        #     test_data = np.transpose(test_data, (0, 3, 1, 2))
        #     test_label_data = np.reshape(test_label_data, (-1, 1))
        #     #             test_data = np.random.uniform(0, 255, (batch_size, 3, Image_Height, Image_Width)).astype(np.float32)
        #     #             test_label_data = np.zeros((5786640 * batch_size, 1)).astype(np.int64).reshape(5786640 * batch_size, 1)
        #     test_cost, test_acc = exe.run(program=test_program,
        #                                   feed={'img': test_data,
        #                                         'label': test_label_data},
        #                                   fetch_list=[avg_cost.name, acc.name])
        #     print("Pass: %d, Batch: %d, Test_Cost: %0.5f, Test_Accuracy: %0.5f" %
        #           (pass_id, batch_id, test_cost[0], test_acc[0]))