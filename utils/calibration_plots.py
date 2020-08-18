# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_uncert(err, sigma, freq_in_bin=None, outlier_freq=0.0):
    if freq_in_bin is not None:
        freq_in_bin = freq_in_bin[torch.where(freq_in_bin > outlier_freq)]  # filter out zero frequencies
        err = err[torch.where(freq_in_bin > outlier_freq)]
        sigma = sigma[torch.where(freq_in_bin > outlier_freq)]
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    max_val = np.max([err.max(), sigma.max()])
    min_val = np.min([err.min(), sigma.min()])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax.plot(sigma, err, marker='.')
    ax.set_ylabel(r'mse')
    ax.set_xlabel(r'uncertainty')
    ax.set_aspect(1)
    fig.tight_layout()
    return fig, ax


def plot_frequency(uncert, in_bin, n_bins=15, range=None):
    if range == None:
        bin_boundaries = np.linspace(0, uncert.max().item(), n_bins + 1)[:-1]
        width = uncert.max().item() / (n_bins * 1.25)
    else:
        bin_boundaries = np.linspace(range[0], range[1], n_bins + 1)[:-1]
        width = range[1] / (n_bins*1.25)
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    plt.bar(bin_boundaries, (in_bin.float() / in_bin.sum()).cpu(), width=width)
    ax.set_ylabel(r'frequency')
    ax.set_xlabel(r'uncertainty')
    fig.tight_layout()
    return fig, ax
