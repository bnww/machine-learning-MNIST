"""This script contains functions for generating binary datasets,
defines a perceptron network architecture and training algorithm"""

"""x is the input layer (784 units), w is the weights for each unit
(each has only 1 weight to the single output node), b is the bias"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

def prepare_binary(x, y, d1, d2): # function for generating binary datasets
    cond = (y == d1) + (y == d2)
    binary_x = x[cond,:]
    binary_y = y[cond]*1.
    print(binary_y[:5])

    binary_y[binary_y == d1] = -1
    binary_y[binary_y == d2] = 1
    return binary_x, binary_y


def predict(x, w, b):
    z = np.sum(x*w) + b
    activation = (1 - np.exp(-z)) / (1 + np.exp(-z)) # TanH
    return 1 if activation >0 else -1


def calc_loss(y_, y): # class dissimilarity loss function
    if type(y) == np.ndarray:
        errors = list()
        for i in range(len(y)):
            errors.append(0) if y_[i] == y[i] else errors.append(1)
        loss = np.sum(errors)/len(y)
    else:
        loss = 0 if y_ == y else 1
    return loss


def optimize(x, y, initial_l_rate = 1, verbose = False):  # x is the training dataset, y are the true values
    iter = 0
    loss = np.inf
    n, m = x.shape
    w = 2*(np.random.rand(m)) - 1 # scale random values to [-1,1)
    b = 0
    losses = {}
    while (iter <= 1000) & (loss > 1e-3):
        y_ = [predict(x[i], w, b) for i in range(n)]
        loss = calc_loss(y_, y)
        losses[iter] = loss # calculating loss for current iteration of training

        l_rate = initial_l_rate - (np.floor(iter/100)*0.1) # step decay - l rate decays by 0.1 every 100 iterations.
        if verbose:
            print(f'\nIteration {iter}\nloss: {loss}\nL_rate: {l_rate}')

        iter += 1
        sample = np.random.randint(n) # select single sample for training
        sample_pred = predict(x[sample], w, b)
        sample_loss = calc_loss(sample_pred, y[sample])
        if sample_loss == 0: # don't update weights if prediction for sample is correct
            continue
        else:
            for i in range(len(w)): # update weights for each pixel
                w[i] = w[i] + (l_rate * (y[sample] - sample_pred) * x[sample][i])
            b = b + (l_rate * (y[sample] - sample_pred))
    return w, b, losses


def plot_weights(w, b, d1, d2):
    plt.figure
    final_weights = w+b # adding on bias to each pixel weight
    plt.imshow(final_weights.reshape(28,28), norm = CenteredNorm(0), cmap = 'bwr')
    plt.colorbar()
    plt.title(f'{d1} in blue, {d2} in red')
    plt.savefig(f'weights for {d1}{d2}.png')
