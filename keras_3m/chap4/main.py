import keras
import skeras
import tensorflow.keras as tk
import matplotlib.pyplot as plt
from keras import layers, models
from keras import backend

class CNN(models.Sequential) :
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
        self.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
        self.add(layers.MaxPool2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())

        self.add(layers.Dense(128, activation = 'relu'))
        self.add(layers.Dropout(0.5))

        self.add(layers.Dense(num_classes, activation = 'softmax'))

        self.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

class Data() :
    def __init__(self):
        num_classes = 10

        mnist = tk.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        img_rows, img_cols = X_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255.0
        X_test /= 255.0

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

def main():
    epochs = 10
    batch_size = 128

    data = Data()

    model = CNN(data.input_shape, data.num_classes)
    history = model.fit(data.X_train, data.y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

    score = model.evaluate(data.X_test, data.y_test)
    print('Test loss : ', score[0])
    print('Test accuracy : ', score[1])

    skeras.plot_loss(history)
    plt.show()
    skeras.plot_acc(history)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
