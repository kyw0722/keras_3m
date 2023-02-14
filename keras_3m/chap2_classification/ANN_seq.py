from keras import layers, models

class ANN(models.Sequential) :
    def __init__(self, Nin, Nh, Nout):
        super().__init__()

        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin, )))
        self.add(layers.Dense(Nout, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

if __name__ == '__main__':
    Nin = 8
    Nh = 4
    number_of_class = 2
    Nout = 2

    model = ANN(Nin, Nh, Nout)