from keras.models import load_model
from preprocessing_utils.feature_engineering import CIFAR_10
from sklearn import preprocessing
from keras import models
import numpy as np
import matplotlib.pyplot as plt

'''
函数功能：将每个中间卷积层的所有通道可视化
'''

# 读取模型并输出模型的结构
x_train, _, _, _ = CIFAR_10.load_data()
model = load_model('cifar10_rmsprop_ep250.h5')
model.summary()

# 将训练集归一化，便于输出图像
x_train = x_train.flatten().reshape(-1, 1)
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_train = x_train.reshape(50000, 32, 32, 3)


# 保存卷积层的名称，作为我们图表的标题
layer_names = []
for layer in [model.layers[i] for i in [0, 3, 8, 11, 16, 19]]:
    layer_names.append(layer.name)

# 定义每一个图片集的个数
images_per_row = 16

# 提取出卷积层的输出
layer_outputs = [layer.output for layer in model.layers[:6]]
# 在给定模型输入的情况下，创建将返回这些输出的模型：
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 这将返回5000个Numpy数组的列表：
# 每个数激活一个数组
activations = activation_model.predict(x_train[:5000])

# 显示卷积层图
for layer_name, layer_activation in zip(layer_names, activations):
    # 特征图中的特征个数
    n_features = layer_activation.shape[-1]

    # 特征图的形状为：(1, size, size, n_features)
    size = layer_activation.shape[1]

    # 在这个矩阵中将激活通道平铺
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 将每个过滤器平铺到一个大的水平网格中
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # 对特征进行后处理，使其看起来更美观
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # 显示网格
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()