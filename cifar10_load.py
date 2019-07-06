#CIFAR-10のデータセットのインポート
from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

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
from keras.models import load_model
import numpy as np

model = load_model('./CIFAR-10.h5')

#評価 & 評価結果出力
print(model.evaluate(X_test, Y_test))

#モデルの図示化
from keras.utils import plot_model

plot_model(model, to_file='model.png',show_shapes=True)