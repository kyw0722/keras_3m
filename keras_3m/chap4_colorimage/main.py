import tensorflow.keras as tk
import keras
import aicnn
from keras import backend as K
assert K.image_data_format() == 'channels_last'

class Machine(aicnn.Machine) :
    def __init__(self):
        cifar = tk.datasets.cifar10
        (X, y), (X_test, y_test) = cifar.load_data()
        super().__init__(X, y, nb_classes = 10)

def main() :
    m = Machine()
    m.run()

if __name__ == '__main__' :
    main()