"""This file contains functions for the PCA analysis of the MNIST dataset."""

import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt

"""Standardising the data"""
def standardise_data(data):
    data_s = data / 127.5-1 # max value = 255, so this formula squishes all values to between -1 and 1

    no_features = np.prod(data_s.shape[1:])
    n_data = data_s.shape[0]

    data_s.resize((n_data, no_features))
    print(data_s.shape)
    return data_s


def pca_analysis(data, n_components = 2):
    pca = decomposition.PCA(n_components)
    projected = pca.fit_transform(data)
    print(projected.shape)
    return pca, projected

def plot_pca(projected_data, gt_data):
    plt.scatter(projected_data[:, 0], projected_data[:, 1],
               c=gt_data, alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10), s=1)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
# These points are the projection of each data point along the directions with the largest variance.

def plot_cumulative_explained_variance(data):
    pca = decomposition.PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.grid()


"""The following plot_digits function is taken from Vanderplas, 2016."""
def plot_digits(data, dim=28):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(dim, dim),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
