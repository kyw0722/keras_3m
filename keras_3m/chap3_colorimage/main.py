import keras.callbacks
import numpy as np
import skeras
import matplotlib.pyplot as plt
import tensorflow.keras as tk
from keras import layers, models, optimizers
from keras.utils import np_utils

class DNN(models.Sequential) :
    def __init__(self, Nin, Nh_l, Pd_l, Nout):
        super().__init__()

        for i in range(len(Nh_l)) :
            if i == 0 :
                self.add(layers.Dense(Nh_l[i], activation = 'relu', input_shape = (Nin, )))
                self.add(layers.Dropout(Pd_l[i]))

            else :
                self.add(layers.Dense(Nh_l[i], activation = 'relu'))
                self.add(layers.Dropout(Pd_l[i]))


        # 수정
        # for i in range(len(Nh_l)) :
        #     if i == 0 :
        #         self.add(layers.Dense(Nh_l[i], activation = 'relu', input_shape = (Nin, ), kernel_initializer = 'he_normal'))
        #         self.add(layers.Dropout(Pd_l[i]))
        #
        #     else :
        #         self.add(layers.BatchNormalization())
        #         self.add(layers.Dense(Nh_l[i], activation = 'relu', kernel_initializer = 'he_normal'))
        #         self.add(layers.Dropout(Pd_l[i]))


        self.add(layers.Dense(Nout, activation = 'softmax'))

        self.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(0.001), metrics = ['accuracy'])

def Data_func() :
    cifar = tk.datasets.cifar10

    (X_train, y_train), (X_test, y_test) = cifar.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W * H * C)
    X_test = X_test.reshape(-1, W * H * C)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def main() :
    Nh_l = [100, 50, 25]
    Pd_l = [0.1, 0.1, 0.1]
    number_of_class = 10
    Nout = 10

    (X_train, y_train), (X_test, y_test) = Data_func()

    model = DNN(X_train.shape[1], Nh_l, Pd_l, Nout)
    es = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
    history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, callbacks = [es])

    performance_test = model.evaluate(X_test, y_test, batch_size = 100)
    print('Test Loss and Accuracy ->', performance_test)

    skeras.plot_loss(history)
    plt.show()
    skeras.plot_acc(history)
    plt.show()

if __name__ == '__main__':
    main()