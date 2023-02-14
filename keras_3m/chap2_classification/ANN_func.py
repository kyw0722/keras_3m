from keras import layers, models

class ANN(models.Model) :
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')
        Nh_l = [5, 10, 5]

        hidden_l = []
        for n in Nh_l :
            hidden_l.append(layers.Dense(n))

        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


if __name__ == '__main__':
    Nin = 8
    Nh = 4
    number_of_class = 2
    Nout = 2

    model = ANN(Nin, Nh, Nout)