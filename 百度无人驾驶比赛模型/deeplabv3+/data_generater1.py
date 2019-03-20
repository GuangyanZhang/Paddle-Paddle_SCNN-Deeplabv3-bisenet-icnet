from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import random
import cv2
import numpy as np

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
              txt_l = i.split('.jpg ')

              dic_filelist.append(txt_l[0])
              list_wavmark.append(txt_l[1])
              #j = j + 1
              # print(txt_l[0])
              #print(list_wavmark[str(j)])

      txt_obj.close()
      return dic_filelist, list_wavmark


def get_data(Height, Width,train_image,label_image):
    # train_image = dic_filelist[num]
    # train_image = list_wavmark[num]
    # print('train_image',train_image)
    train_data = cv2.imread(train_image+'.jpg')
    #     train_data = PIL.Image.open(train_image)
    image = resize_image(train_data, Height, Width)
    # print('image：', image.shape)
    # train_data = fluid.layers.resize_bilinear(train_data, out_shape=[1024,512])
    label_image = cv2.imread(label_image)
    # print('label_image',label_image)
    # print('label_image_before：', label_image.shape)
    label_image = resize_image(label_image, Height, Width)
    # print('label_image_after：', label_image.shape)
    #     label_image = PIL.Image.open(label_image)
    return image, label_image[:, :, 0] # 取了一个通道


"""
批量读取图片
"""

def data_generator(Height, Width, batch_size,dic_filelist,list_wavmark):
    X = np.zeros((batch_size, Height, Width, 3), dtype=np.float32)  # batch_size train data
    Y = np.zeros((batch_size, Height, Width), dtype=np.int32)  # batch_size label data

    for i in range(batch_size):
        # num = i -batch_id*2
        num = random.randint(0, 66966)
        if num in dic_filelist:
            pass
        else:

            num = random.randint(0, 66966)
        # num = 34722
        # print("number ",num)

        #ran_num = random.randint(0, 3999) #生成一个随机数
        train_data, label_image = get_data(Height, Width,dic_filelist[num],list_wavmark[num])
        # print('train_data：', train_data.shape)
        # print('label_image：', label_image.shape)
        X[i, 0:len(train_data)] = train_data
        Y[i,0:len(label_image)] = label_image
        #X = X[:, :, :, ::-1].astype(np.float32) / (255.0 / 2) - 1
        # Y = Y.astype(np.int32)[:, :, :, 0]

        # print(Y.shape)
    return X, Y
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
    colorize[8,:] = 205

    ims = colorize[gts,:].reshape([gts.shape[0],gts.shape[1],1])
    return ims
