#CIFAR-10のデータセットのインポート
from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

image_size = 32

#CIFAR-10の正規化
from keras.utils import to_categorical

# 特徴量の正規化
X_train = X_train/255.
X_test = X_test/255.

# クラスラベルの1-hotベクトル化
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# CNNの構築
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.applications import VGG16
import numpy as np

# VGG16の構築
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = Sequential()
model.add(vgg_conv)
 
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

#訓練
history = model.fit(X_train, Y_train, epochs=20)

# モデルの保存
model.save('./CIFAR10_VGG16.h5')

#評価 & 評価結果出力
print(model.evaluate(X_test, Y_test))

#モデルの図示化
from keras.utils import plot_model

plot_model(model, to_file='model.png',show_shapes=True)