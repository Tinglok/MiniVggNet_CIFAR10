from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np

model = load_model('cifar10_rmsprop_ep250.h5')
model.summary()  # As a reminder.
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

# 输出预测准确率
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))