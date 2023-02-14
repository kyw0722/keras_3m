import numpy as np
import matplotlib.pyplot as plt
import os

def save_history_history(fname, history_history, fold='') :
    np.save(os.path.join(fold, fname), history_history)

def load_history_history(fname, fold='') :
    history_history = np.load(os.path.join(fold, fname)).item(0)

    return history_history

def plot_loss(history) :
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 0)

def plot_acc(history) :
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 0)