import keras.callbacks
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tk
from keras import layers, models
from keras.utils import np_utils

class DNN(models.Sequential) :
    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()

        for i in range(len(Nh_l)) :
            if i == 0 :
                self.add(layers.Dense(Nh_l[i], activation = 'relu', input_shape = (Nin, )))
                self.add(layers.Dropout(0.2))

            else :
                self.add(layers.Dense(Nh_l[i], activation = 'relu'))
                self.add(layers.Dropout(0.2))

        self.add(layers.Dense(Nout, activation = 'softmax'))

        self.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

def Data_func() :
    mnist = tk.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def main() :
    Nin = 784
    Nh_l = [100, 50, 25, 10]
    number_of_class = 10
    Nout = number_of_class

    (X_train, y_train), (X_test, y_test) = Data_func()

    model = DNN(Nin, Nh_l, Nout)

    es = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
    history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, verbose = 2, callbacks = [es])

    performance_test = model.evaluate(X_test, y_test, batch_size = 100)
    print('Test Loss and Accuracy ->', performance_test)

if __name__ == '__main__':
    main()