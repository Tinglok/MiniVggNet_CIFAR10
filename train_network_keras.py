from preprocessing_utils.feature_engineering import CIFAR_10
from network_model.minivggnet_keras import MiniVGGNetKeras
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
import keras.backend as K

# 函数以epoch为参数（从0算起的整数），返回原学习率的一半
def scheduler(epoch):
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

# 读取CIFAR-10数据集，并进行基本的数据预处理
x_train, y_train, x_test, y_test = CIFAR_10.load_data()

# 调用数据增强方法，增加图片的多样性，有利于增强模型的泛化程度
datagen = CIFAR_10.data_augumentation()
datagen.fit(x_train)

'''
利用rmsprop优化梯度下降
前100个epoch学习率设为0.001
100-200个epoch学习率设为0.0005
200-250个epoch学习率设为0.00025
'''
batch_size = 64
epochs = 25

# 调用预设的模型结构
model = MiniVGGNetKeras.creat_model()

# 创建一个计算学习率对象
reduce_lr = LearningRateScheduler(scheduler)

# 建立反馈参数
callback_list = [
    reduce_lr,
    TensorBoard(log_dir='./Graph')      # 使用tensorboard，保存训练集和验证集的acc、loss数据，并自动生成网络拓扑图
]

# 保存前250个epoch模型的权值
opt_rms = optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=10*epochs,
    verbose=1,
    callbacks=callback_list,
    validation_data=(x_test,y_test))
model.save('cifar10_rmsprop_ep250.h5')

# 输出预测准确率
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))