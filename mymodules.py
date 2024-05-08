import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh
import numpy as np

from numpy.typing import NDArray
from typing import Tuple, Optional, Union
from mytypes import Mask, Filename


### data
def is_real(photon):
    '''returns True for a real photon and False for a fake'''
    try:
        pdgId = photon.genParticle().pdgId()
        if pdgId == 22:
            return True  # real
        else:
            return False  # fake
    except ReferenceError:
        return False  # fake


##################################
### helper
def get_centered_binning(binwidth: float, start: float, end: float) -> Tuple[int, float, float]:
    num_bins = int(abs(end-start)/binwidth) + 1
    binning = (num_bins, start-binwidth/2, end+binwidth/2)
    return binning


###################################
### plotting
def histplot(ax: plt.Axes, data: NDArray, binning: Tuple[int, float, float],
             normalizer: Optional[Union[bool, int, float, NDArray]] = None,
             weights: Optional[NDArray] = None,
             **kwargs) -> bh.Histogram:
    """
    put data in a boosthist with binning for bh.Axes.Regular and plot it along the giving axis
    data is normalized to normalizer: int or summable, is set to 1 if False or set to hist.sum() if None
    kwargs are passed to plot
    returns the histogram
    """
    hist = bh.Histogram(bh.axis.Regular(*binning))
    hist.fill(data)
    if weights is not None:
        hist *= weights
    if normalizer is None:
        normalizer = hist.sum()
    elif normalizer is False:
        normalizer = 1
    elif isinstance(normalizer, bh.Histogram):
        normalizer = normalizer.sum()
    else:
        normalizer = bh.Histogram(bh.axis.Regular(*binning)).fill(normalizer).sum()

    values = hist.view() / normalizer
    error = np.sqrt(hist.view()) / normalizer
    ax.errorbar(hist.axes[0].centers, values, yerr=error, ds='steps-mid', **kwargs)
    return hist

def plot_dist(df: pd.DataFrame, data: NDArray,
              binning: Tuple[int, float, float],
              mask: Optional[Mask] = None,
              yscale: Optional[str] = 'log',
              xlabel: Optional[str] = None,
              savename: Optional[str] = None,
              **kwargs) -> plt.Figure:
    """plots a (individually) normalized distribution of real and fake split into barrel and endcap
    if mask is given, there will be two more subplots with the mask applied
    all kwargs are passed to histplot"""
    barrel = df.detID
    endcap = ~barrel
    real = df.real
    fake = ~real

    rows, cols = (1,2) if mask is None else (2,2)
    size = (10,4) if mask is None else (10,8)
    fig, axs = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    axs = axs.flat
    histplot(axs[0], data[barrel & real], binning, color='blue', label='real', **kwargs)
    histplot(axs[0], data[barrel & fake], binning, color='red', label='fake', **kwargs)
    histplot(axs[1], data[endcap & real], binning, color='blue', label='real', **kwargs)
    histplot(axs[1], data[endcap & fake], binning, color='red', label='fake', **kwargs)
    axs[0].set_title('Barrel')
    axs[1].set_title('Endcap')
    if mask is not None:
        histplot(axs[2], data[barrel & real][mask], binning, color='blue', label='real',  **kwargs)
        histplot(axs[2], data[barrel & fake][mask], binning, color='red', label='fake', **kwargs)
        histplot(axs[3], data[endcap & real][mask], binning, color='blue', label='real', **kwargs)
        histplot(axs[3], data[endcap & fake][mask], binning, color='red', label='fake', **kwargs)
        axs[0].set_ylabel('without cuts')
        axs[2].set_ylabel('with cuts')

    for ax in axs:
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.legend()

    if savename is not None:
        fig.savefig(savename)
    return fig

def plot_subset(ax: plt.Axes, data: NDArray, subset_mask: Mask, binning: Tuple[int, float, float],
                colors: Tuple[Optional[str], Optional[str], Optional[str]] = ('black', 'blue', 'red'),
                labels: Tuple[Optional[str], Optional[str], Optional[str]] = ('All', 'Subset1', 'Subset2'),
                **kwargs) -> None:
    """
    plots the data, its subset and the opposite set with a given binning on the given axis
    subsets get normalized to data
    if no colors given: uses black, blue, red
    kwargs are passed to all three plots
    """
    subset1 = data[subset_mask]
    subset2 = data[~subset_mask]

    histplot(ax, data, binning, normalizer=None, color=colors[0], label=labels[0], **kwargs)
    histplot(ax, subset1, binning, normalizer=data, color=colors[1], label=labels[1], **kwargs)
    histplot(ax, subset2, binning, normalizer=data, color=colors[2], label=labels[2], **kwargs)

def plot_accuracy(ax: plt.Axes, history) -> None:
    ax.plot(history['accuracy'], label='train_accuracy')
    ax.plot(history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')

def plot_loss(ax: plt.Axes, history) -> None:
    ax.plot(history['loss'], label='train_loss')
    ax.plot(history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')

def plot_training(history, test_acc: Optional[float] = None, savename: Optional[str] = None) -> plt.Figure:
    '''plot loss and accuracy,
    puts test_acc in title if given
    creates and returns fig
    saves fig if savename is given'''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    if test_acc is not None:
        plt.title(f'test accuracy = {test_acc: .4}')
    plot_loss(ax1, history)
    plot_accuracy(ax2, history)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
        print('INFO: training plot saved as:', savename)
    return fig

def plot_output(ax: plt.Axes, y_pred: NDArray, real: Mask, binning: Tuple[int, float, float]) -> None:
    histplot(ax, y_pred[real], binning, color='blue', label='real')
    histplot(ax, y_pred[~real], binning, color='red', label='fake')
    ax.legend(loc='upper center')
    ax.set_xlabel('Output score')
    ax.set_ylabel('#')
    ax.grid(True)


