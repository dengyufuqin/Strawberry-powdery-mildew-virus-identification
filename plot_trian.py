import numpy as np
from data_utils import DataSet
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from ResNet50 import ResNet50
import numpy as np
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline

# with open('trainHistoryDict_6.txt', 'rb') as file_pi:
# #     history1=pickle.load(file_pi)
# # with open('trainHistoryDict_7.txt', 'rb') as file_pi:
# #     history2=pickle.load(file_pi)
# # with open('trainHistoryDict_8.txt', 'rb') as file_pi:
# #     history3=pickle.load(file_pi)
# # with open('trainHistoryDict_9.txt', 'rb') as file_pi:
# #     history4=pickle.load(file_pi)
#
#
# acc1 = history1['acc']
# val_acc1 = history1['val_acc']
# acc2 = history2['acc']
# val_acc2 = history2['val_acc']
# acc3 = history3['acc']
# val_acc3 = history3['val_acc']
# acc4 = history4['acc']
# val_acc4 = history4['val_acc']
#
# fig = plt.figure(figsize=(16, 8))
# ax1 = fig.add_subplot(121) # 1x1 第一个图
# ax1.plot(acc1, label='train: test = 6 : 4')
# ax1.plot(acc2, label='train: test = 7 : 3')
# ax1.plot(acc3, label='train: test = 8 : 2')
# ax1.plot(acc4, label='train: test = 9 : 1')
#
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Accuracy')
# ax1.set_title('Different training set test set proportions of Train')
# ax1.legend()
#
#
# ax2 = fig.add_subplot(122)
# ax2.plot(val_acc1, label='train: test = 6 : 4')
# ax2.plot(val_acc2, label='train: test = 7 : 3')
# ax2.plot(val_acc3, label='train: test = 8 : 2')
# ax2.plot(val_acc4, label='train: test = 9 : 1')
#
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Accuracy')
# ax2.set_title('Different training set test set proportions of Test')
# ax2.legend()
#
#
# plt.tight_layout()
# plt.show()


# with open('trainHistoryDict_6.txt', 'rb') as file_pi:
#     history1=pickle.load(file_pi)
# with open('trainHistoryDict_7.txt', 'rb') as file_pi:
#     history2=pickle.load(file_pi)
# with open('trainHistoryDict_8.txt', 'rb') as file_pi:
#     history3=pickle.load(file_pi)
# with open('trainHistoryDict_9.txt', 'rb') as file_pi:
#     history4=pickle.load(file_pi)
#
#
# with open('trainHistoryDict_adam_1.txt', 'rb') as file_pi:
#     history2=pickle.load(file_pi)
# with open('trainHistoryDict_adam_2.txt', 'rb') as file_pi:
#     history3=pickle.load(file_pi)
# with open('trainHistoryDict_adam_3.txt', 'rb') as file_pi:
#     history4=pickle.load(file_pi)
#
# acc2 = history2['acc']
# val_acc2 = history2['val_acc']
# acc3 = history3['acc']
# val_acc3 = history3['val_acc']
# acc4 = history4['acc']
# val_acc4 = history4['val_acc']
#
# fig = plt.figure(figsize=(16, 8))
# ax1 = fig.add_subplot(121) # 1x1 第一个图
# ax1.plot(acc2, label='adam = 0.01')
# ax1.plot(acc3, label='adam = 0.001')
# ax1.plot(acc4, label='adam = 0.0001')
#
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Accuracy')
# ax1.set_title('Different learning_rate of Train')
# ax1.legend()
#
#
# ax2 = fig.add_subplot(122)
# ax2.plot(val_acc2, label='adam = 0.01')
# ax2.plot(val_acc3, label='adam = 0.001')
# ax2.plot(val_acc4, label='adam = 0.0001')
#
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Accuracy')
# ax2.set_title('Different learning_rate of Test')
# ax2.legend()
#
#
# plt.tight_layout()
# plt.show()

with open('trainHistoryDict_l.txt', 'rb') as file_pi:
    history1=pickle.load(file_pi)
with open('trainHistoryDict_9.txt', 'rb') as file_pi:
    history2=pickle.load(file_pi)
with open('trainHistoryDict_Update.txt', 'rb') as file_pi:
    history3=pickle.load(file_pi)
with open('trainHistoryDict.txt', 'rb') as file_pi:
    history4=pickle.load(file_pi)

acc1 = history1['acc']
val_acc1 = history1['val_acc']
acc2 = history2['acc']
val_acc2 = history2['val_acc']
acc3 = history3['acc']
val_acc3 = history3['val_acc']
acc4 = history4['acc']
val_acc4 = history4['val_acc']

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121) # 1x1 第一个图
ax1.plot(acc2, label='no change')
ax1.plot(acc3, label='add L2 and lambda=0.01')
ax1.plot(acc4, label='change the activation')
ax1.plot(acc1, label='add L2 and lambda=0.000001')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Change the network of Train')
ax1.legend()


ax2 = fig.add_subplot(122)
ax2.plot(val_acc2, label='no change')
ax2.plot(val_acc3, label='add L2 and lambda=0.01')
ax2.plot(val_acc4, label='change the activation')
ax2.plot(val_acc1, label='add L2 and lambda=0.000001')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Change the network of Test')
ax2.legend()


plt.tight_layout()
plt.show()
