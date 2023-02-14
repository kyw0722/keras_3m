import numpy as np
import skeras
import tensorflow.keras as tk
import matplotlib.pyplot as plt
from keras import layers, models

class AE(models.Model) :
    def __init__(self, x_nodes, h_dim) :
        x_shapes = (x_nodes, )

        x = layers.Input(shape = x_shapes)
        h = layers.Dense(h_dim, activation = 'relu')(x)
        # h = layers.Dense(h_dim, activation='relu')(h)
        y = layers.Dense(x_nodes, activation = 'sigmoid')(h)

        super().__init__(x, y)

        self.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        self.x = x
        self.h = h
        self.h_dim = h_dim

    def Encoder(self) :
        return models.Model(self.x, self.h)

    def Decoder(self) :
        h_shape = (self.h_dim, )
        h = layers.Input(shape = h_shape)
        output_layer = self.layers[-1]
        y = output_layer(h)

        return models.Model(h, y)

def main():
    mnist = tk.datasets.mnist
    (X_train, _), (X_test, _) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))

    x_nodes = 784
    h_dim = 36

    autoencoder = AE(x_nodes, h_dim)

    history = autoencoder.fit(X_train, X_train, epochs = 100, batch_size = 256, shuffle = True, validation_data = (X_test, X_test))

    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    skeras.plot_acc(history)
    plt.show()
    skeras.plot_loss(history)
    plt.show()

    n = 10
    plt.figure(figsize = (20, 6))

    for i in range(n) :
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == '__main__':
    main()
