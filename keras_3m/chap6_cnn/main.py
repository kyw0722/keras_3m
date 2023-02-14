import tensorflow.keras as tk
import matplotlib.pyplot as plt
from keras import layers, models, backend, utils
from skeras import plot_loss, plot_acc

def Conv2D(filters ,kernel_size, padding = 'same', activation = 'relu') :
    return layers.Conv2D(filters, kernel_size, padding = padding, activation = activation)

class AE(models.Model) :
    def __init__(self, org_shape = (1, 28, 28)) :
        original = layers.Input(shape = org_shape)

        conv2d1 = Conv2D(4, (3, 3))(original)
        pool1 = layers.MaxPooling2D((2, 2), padding = 'same')(conv2d1)

        conv2d2 = Conv2D(8, (3, 3))(pool1)
        pool2 = layers.MaxPooling2D((2, 2), padding = 'same')(conv2d2)

        conv2d3 = Conv2D(1, (7, 7))(pool2)

        conv2d4 = Conv2D(16, (3, 3))(conv2d3)
        upsample1 = layers.UpSampling2D((2, 2))(conv2d4)

        conv2d5 = Conv2D(8, (3, 3))(upsample1)
        upsample2 = layers.UpSampling2D((2, 2))(conv2d5)

        conv2d6 = Conv2D(4, (3, 3))(upsample2)

        decoded = Conv2D(1, (3, 3), activation = 'sigmoid')(conv2d6)

        super().__init__(original, decoded)
        self.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

class Data :
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

        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

def show_ae(autoencoder, data) :
    X_test = data.X_test
    decoded_imgs = autoencoder.predict(X_test)

    if backend.image_data_format() == 'channels_first' :
        N, n_ch, n_i, n_j = X_test.shape
    else :
        N, n_i, n_j, n_ch = X_test.shape

    X_test = X_test.reshape(N, n_i, n_j)
    decoded_imgs = decoded_imgs.reshape(decoded_imgs.shape[0], n_i, n_j)

    n = 10
    plt.figure(figsize = (20, 4))

    for i in range(n) :
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i], cmap = 'gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def main(epochs = 20, batch_size = 128):
    data = Data()
    autoencoder = AE(data.input_shape)

    history = autoencoder.fit(data.X_train, data.X_train, epochs = epochs, batch_size = batch_size, shuffle = True, validation_split = 0.2)

    plot_acc(history)
    plt.show()
    plot_loss(history)
    plt.show()

    show_ae(autoencoder, data)
    plt.show()

if __name__ == '__main__':
    main()
