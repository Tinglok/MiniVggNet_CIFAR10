from preprocessing_utils.feature_engineering import CIFAR_10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers


'''
本模型是基于VGG16修改简化而来的
卷积核：使用固定尺寸的 3×3
卷积层：使用same填充方式，即使得卷积前后尺寸相同
激活函数：使用ELU，实际训练效果比ReLu好
池化层：使用 2×2 Maxpooling
使用L2正则化,正则化系数为1e-4
类比使用 VGG16 的前三个卷积结构：以2的幂次递增卷积核数量 (32, 64, 128)
卷积层输出后直接输入 10 分类的 Softmax Classifier
测试过添加一层128个节点的全连接层节点在 Softmax 层之前，但是训练250次只能到79%的准确率，比不加还要差，故舍弃
模型预测精度在91%左右
'''
class MiniVGGNetKeras:
    def creat_model():
        # 读取CIFAR_10数据集中的x_train，用它的shape作为输入
        x_train, _, _, _ = CIFAR_10.load_data()
        # 设置L2正则化系数和数据集中的类别数
        weight_decay = 1e-4
        num_classes = 10

        # first (CONV => RELU) * 2 => POOL layer set
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        # second (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        # third (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        # Softmax 分类器
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

        # 输出模型结构
        model.summary()

        # 返回模型结构
        return model