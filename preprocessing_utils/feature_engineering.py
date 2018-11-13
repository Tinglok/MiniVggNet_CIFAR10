from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class CIFAR_10:
    '''
    读取数据并加工，数据类型如下：
    X_train.shape = (50000, 32, 32, 3)
    X_test.shape = (10000, 32, 32, 3)
    y_train.shape = (50000, 1)
    y_test.shape = (10000, 1)
    故在输入模型时需要舍弃shape[0]
    '''
    def load_data():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # 将x_train，x_test标准化
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

        # 将数据的十类label进行 one-hot 编码，以便后面用 Softmax 分类
        num_classes = 10
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)

        #返回训练集和测试集
        return x_train, y_train, x_test, y_test


    # 数据增强，即将图片进行翻转或者移动，有利于模型的泛化
    def data_augumentation():
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )

        # 返回数据增强生成器
        return datagen