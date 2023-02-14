from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from keras import layers, models

class Data :
    def __init__(self, max_features = 20000, maxlen = 80) :
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features)

        X_train = pad_sequences(X_train, maxlen = maxlen)
        X_test = pad_sequences(X_test, maxlen = maxlen)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

class RNN_LSTM(models.Model) :
    def __init__(self, max_features, maxlen):
        x = layers.Input((maxlen,))
        embedding = layers.Embedding(max_features, 128)(x)
        Bidirection = layers.Bidirectional(layers.LSTM(128, dropout = 0.2))(embedding)
        # LSTM = layers.LSTM(128, dropout = 0.2, recurrent_dropout = 0.2)(embedding)
        y = layers.Dense(1, activation = 'sigmoid')(Bidirection)

        super().__init__(x, y)

        self.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

class Machine :
    def __init__(self, max_features = 20000, maxlen = 80) :
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    def run(self, epochs = 3, batch_size = 32) :
        data = self.data
        model = self.model
        print('Training stage')
        print('=================')

        model.fit(data.X_train, data.y_train, batch_size = batch_size, epochs = epochs, validation_data = (data.X_test, data.y_test))

        score, acc = model.evaluate(data.X_test, data.y_test, batch_size = batch_size)
        print('Test performance : accuracy = {0}, loss = {1}'.format(acc, score))

def main() :
    m = Machine()
    m.run()

if __name__ == '__main__':
    main()