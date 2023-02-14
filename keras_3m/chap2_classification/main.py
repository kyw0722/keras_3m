# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tk
from keras import layers, models
from keras.utils import np_utils

def make_functional_ANN(Nin, Nh, Nout) :
    x = layers.Input(shape = (Nin,))
    h = layers.Activation('relu')(layers.Dense(Nh)(x))
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))

    model = models.Model(x, y)

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def make_sequential_ANN(Nin, Nh, Nout) :
    model = models.Sequential()

    model.add(layers.Dense(Nh, activation = 'relu', input_shape = (Nin,)))
    model.add(layers.Dense(Nout, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def plot_loss(history) :
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 0)

def plot_acc(history) :
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class

    mnist = tk.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, H, W = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = make_sequential_ANN(Nin, Nh, Nout)
    history = model.fit(X_train, Y_train, epochs = 15, batch_size = 100, validation_split = 0.2)

    performance_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy -> {:.2f}, {:.2f}'.format(*performance_test))

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()