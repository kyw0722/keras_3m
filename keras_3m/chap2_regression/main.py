import tensorflow.keras as tk
import skeras
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn import preprocessing

class ANN(models.Model) :
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')

        x = layers.Input(shape = (Nin,))
        h = relu(hidden(x))
        y = output(h)

        super().__init__(x, y)
        self.compile(loss = 'mse', optimizer = 'adam')

def Data_func() :
    boston = tk.datasets.boston_housing
    (X_train, y_train), (X_test, y_test) = boston.load_data()
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_test, y_test)

def main() :
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()

    history = model.fit(X_train, y_train, epochs=300, batch_size=100, validation_split=0.2, verbose=2)
    performance_test = model.evaluate(X_test, y_test, batch_size = 100)
    print('\nTest Loss -> {:.2f}'.format(performance_test))

    skeras.plot_loss(history)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()