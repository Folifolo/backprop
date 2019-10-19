from keras.datasets import mnist
from keras.utils import np_utils
from Network import NN
import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = NN(784, 50, 10, 0.1, 1000)
    model.init_weights()
    model.fit(x_train, y_train, 100, (x_test, y_test))


