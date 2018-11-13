from keras import backend as K
from keras.models import load_model
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
model = load_model('cifar10_rmsprop_ep250.h5')

# 将张量转换为有效图像的实用函数
def deprocess_image(x):
    # 对张量作标准化，使其均值为0，标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 将x裁切(clip)到[0,1]区间
    x += 0.5
    x = np.clip(x, 0, 1)

    # 将x转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 生成过滤器可视化的函数
def generate_pattern(layer_name, filter_index, size=150):
    # 构建一个损失函数，将盖层第n个过滤器的激活最大化
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 计算损失相对于输入图像的梯度
    grads = K.gradients(loss, model.input)[0]

    # 标准化技巧：将梯度标准化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 返回给定输入图像的损失和梯度
    iterate = K.function([model.input], [loss, grads])

    # 从带有噪声的灰度图像开始
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 运行40次梯度上升
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

for layer_name in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']:
    size = 64
    margin = 5

    # 空图像（全黑色），用于保存结果
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(4):  # 遍历results网格的行
        for j in range(4):  # 遍历results网格的列
            # 生成layer_name层第i+(j*8)个过滤器的模式
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # 将结果放到results网格第(i,j)个方块中
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # 将结果归一化，便于绘制图像
    results = results.flatten().reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    results = min_max_scaler.fit_transform(results)
    results = results.reshape(547, 547, 3)
    # 显示results网格
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()