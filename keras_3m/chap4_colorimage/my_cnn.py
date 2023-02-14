from tensorflow import keras
from keras import models, layers
from keras import datasets
from keras import utils
from sklearn.preprocessing import MinMaxScaler

class DataSet :
    def __init__(self, nb_classes = 10) :
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        scaler = MinMaxScaler()
        n = X_train.shape[0]
        X_train = scaler.fit_transform(X_train.reshape(n, -1)).reshape(X_train.shape)
        n = X_test.shape[0]
        X_test = scaler.transform(X_test.reshape(n, -1)).reshape(X_test.shape)
        self.scaler = scaler

        y_train = utils.to_categorical(y_train, nb_classes)
        y_test = utils.to_categorical(y_test, nb_classes)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test


class CNN(models.Model) :
    def __init__(self, in_shape = None, nb_classes = 10) :

        input1 = layers.Input(in_shape)
        conv2d1 = layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu')(input1)
        pooling1 = layers.MaxPooling2D(pool_size = (2, 2))(conv2d1)
        dropout1 = layers.Dropout(0.2)(pooling1)
        flatten1 = layers.Flatten()(dropout1)

        input2 = layers.Input(in_shape)
        conv2d2 = layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu')(input2)
        pooling2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2d2)
        dropout2 = layers.Dropout(0.2)(pooling2)
        flatten2 = layers.Flatten()(dropout2)

        input3 = layers.Input(in_shape)
        conv2d3 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input3)
        pooling3 = layers.MaxPooling2D(pool_size=(2, 2))(conv2d3)
        dropout3 = layers.Dropout(0.2)(pooling3)
        flatten3 = layers.Flatten()(dropout3)

        concat = layers.Concatenate()([flatten1, flatten2, flatten3])
        normal = layers.BatchNormalization()(concat)
        relu = layers.Dense(16, activation = 'relu')(normal)
        dropout = layers.Dropout(0.2)(relu)

        output = layers.Dense(nb_classes, activation = 'softmax')(dropout)

        super().__init__([input1, input2, input3], output)

        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

def main() :
    data = DataSet()

    model = CNN(in_shape = data.X_train.shape[1:])
    es = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min')
    model.fit([data.X_train, data.X_train, data.X_train], data.y_train, epochs = 10, validation_split = 0.2, callbacks = es)

    score = model.evaluate([data.X_test, data.X_test, data.X_test], data.y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__' :
    main()