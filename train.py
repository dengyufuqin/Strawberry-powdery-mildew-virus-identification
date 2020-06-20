import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
import matplotlib.pyplot as plt
from vis import train_vis
from data_utils import DataSet
import random
from up_ResNet import ResNet50_1
from ResNet50 import ResNet50
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.models import load_model
from VGG16 import VGG_16
from keras.callbacks import TensorBoard
import pickle
from keras import regularizers

seed = 7
np.random.seed(seed)
X, Y, X_test, Y_test = DataSet()
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=random.randint(0, 100))
# _, test_X, _, test_Y = train_test_split(X, Y, test_size=0.5, random_state=random.randint(0, 100))

reg = regularizers.l2(1e-6)
model = ResNet50_1(input_shape=(128, 128, 3), classes=2, reg=reg)
adam = Adam(lr=1e-4, decay=1e-7)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])


# 普通的训练
history = model.fit(X_train, Y_train, epochs=200, batch_size=32,
          validation_data=(X_valid, Y_valid))

preds = model.evaluate(X_test, Y_test)

print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))


with open('trainHistoryDict_perfect.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


print(history)
train_vis(history)


