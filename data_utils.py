import numpy as np
import random
from keras.preprocessing import image
import os


def DataSet():
    # 首先需要定义训练集和测试集的路径，这里创建了trian，和test文件夹
    # 每个文件夹下又创建了文件夹
    train_path_health = './train/health'
    train_path_powder = './train/powder_mideqw'

    test_path_health = './test/health'
    test_path_powder = './test/powder_mideqw'

    train_imglist_health = os.listdir(train_path_health)
    train_imglist_powder = os.listdir(train_path_powder)

    test_imglist_health = os.listdir(test_path_health)
    test_imglist_powder = os.listdir(test_path_powder)

    X_train = np.empty((len(train_imglist_health)+len(train_imglist_powder), 128, 128, 3))
    Y_train = np.empty((len(train_imglist_health)+len(train_imglist_powder), 2))

    X_test = np.empty((len(test_imglist_health) + len(test_imglist_powder), 128, 128, 3))
    Y_test = np.empty((len(test_imglist_health) + len(test_imglist_powder), 2))

    # count对象用来计数，每添加一张图片便+1
    count = 0

    # 遍历/train/health_f即训练集下所有健康正面的图片
    for img_name in train_imglist_health:
        # 得到图片的路径
        img_path = train_path_health + '/' + img_name

        # 通过img.load_img()函数读取对应图片，并转换成目标大小
        # image是tensorflow.keras.preprocessing中的一个对象
        img = image.load_img(img_path, target_size=(128, 128))
        # 将图片转换成numpy数组，并处以255归一化
        # 转化成img的shape是(128, 128, 3)
        img = image.img_to_array(img) / 255.0
        # 将处理好的图片装进定义好的X_train对象中
        X_train[count] = img
        Y_train[count] = np.array((1, 0))
        count += 1

    # print(count)

    for img_name in train_imglist_powder:
        # 得到图片的路径
        img_path = train_path_powder + '/' + img_name

        # 通过img.load_img()函数读取对应图片，并转换成目标大小
        # image是tensorflow.keras.preprocessing中的一个对象
        img = image.load_img(img_path, target_size=(128, 128))
        # 将图片转换成numpy数组，并处以255归一化
        # 转化成img的shape是(128, 128, 3)
        img = image.img_to_array(img) / 255.0
        # 将处理好的图片装进定义好的X_train对象中
        X_train[count] = img
        Y_train[count] = np.array((0, 1))
        count += 1

    # count对象用来计数，每添加一张图片便+1
    count = 0
    # 遍历/train/health_f即训练集下所有健康正面的图片
    for img_name in test_imglist_health:
        # 得到图片的路径
        img_path = test_path_health + '/' + img_name

        # 通过img.load_img()函数读取对应图片，并转换成目标大小
        # image是tensorflow.keras.preprocessing中的一个对象
        img = image.load_img(img_path, target_size=(128, 128))
        # 将图片转换成numpy数组，并处以255归一化
        # 转化成img的shape是(128, 128, 3)
        img = image.img_to_array(img) / 255.0
        # 将处理好的图片装进定义好的X_train对象中
        X_test[count] = img
        Y_test[count] = np.array((1, 0))
        count += 1

    for img_name in test_imglist_powder:
        # 得到图片的路径
        img_path = test_path_powder + '/' + img_name
        # 通过img.load_img()函数读取对应图片，并转换成目标大小
        # image是tensorflow.keras.preprocessing中的一个对象
        img = image.load_img(img_path, target_size=(128, 128))
        # 将图片转换成numpy数组，并处以255归一化
        # 转化成img的shape是(128, 128, 3)
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 1))
        count += 1

    # 打乱训练集中的数据
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]


    return X_train, Y_train, X_test, Y_test
