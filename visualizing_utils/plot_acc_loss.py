import pandas as pd
import matplotlib.pyplot as plt

# 读取tensorboard保存的acc、loss数据
df_acc = pd.read_csv('run_.-tag-acc.csv')
df_loss = pd.read_csv('run_.-tag-loss.csv')
df_val_acc = pd.read_csv('run_.-tag-val_acc.csv')
df_val_loss = pd.read_csv('run_.-tag-val_loss.csv')
acc = df_acc['Value']
loss = df_loss['Value']
val_acc = df_val_acc['Value']
val_loss = df_val_loss['Value']

# 本次实验一共训练250个epoch
epochs = 250

# 绘制训练的acc曲线图
plt.plot(range(epochs), acc, label='Training acc')
plt.plot(range(epochs), val_acc, label='Validation acc')
plt.plot(range(epochs), loss, '-.', label='Training loss')
plt.plot(range(epochs), val_loss, '-.', label='Validation loss')

# 绘制标题
plt.title('Training and validation accuracy & loss')
# 绘制标签
plt.legend()
# 输出图像
plt.show()