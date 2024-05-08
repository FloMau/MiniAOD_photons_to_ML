import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import boost_histogram as bh
import functools
import operator
import pickle
import myplotparams

from typing import Optional, Tuple, Union
from mytypes import Mask, NDArray, Filename

#########################################################
### plot 1d reweighted dists as crosschecks
def plot_hist(ax_: plt.Axes, hist: bh.Histogram, normalize: bool = True, **kwargs) -> None:
    normalizer: float = hist.sum() if normalize else 1.
    ax_.plot(hist.axes[0].edges[:-1], hist.view() / normalizer, ds='steps-post', **kwargs)
    ax_.grid(True)
    ax_.legend(loc='upper right')
    ax_.set_ylabel('#')
    ax_.set_yscale('log')

def compare_1d(ax_: plt.Axes, hist_axes: bh.axis, data: NDArray, weights_: NDArray, 
               real_: Mask, which: str = 'fake', exclude: Optional[Mask] = None, **kwargs) -> None:
    """compares weighted and unweighted distributions of data in 1d"""
    if exclude is None: exclude = np.ones(data.shape, dtype=bool)
    fake_: Mask = ~real_
    real_ = real_ & (~exclude)
    fake_ = fake_ & (~exclude)

    original_real = bh.Histogram(hist_axes)
    original_fake = bh.Histogram(hist_axes)
    weighted_real = bh.Histogram(hist_axes)
    weighted_fake = bh.Histogram(hist_axes)
    original_real.fill(data[real_])
    weighted_real.fill(data[real_], weight=weights_[real_])
    original_fake.fill(data[fake_])
    weighted_fake.fill(data[fake_], weight=weights_[fake_])

    if which=='fake':
        plot_hist(ax_, original_real, label='real', color='blue')
        plot_hist(ax_, original_fake, label='unweighted fake', color='orange', alpha=0.8)
        plot_hist(ax_, weighted_fake, label='weighted fake', ls='--', color='red')
    if which=='real':
        plot_hist(ax_, original_fake, label='fake', color='blue')
        plot_hist(ax_, original_real, label='unweighted real', color='orange', alpha=0.8)
        plot_hist(ax_, weighted_real, label='weighted real', ls='--', color='red')
    ax_.set_ylim(0, None)
    if 'xlabel' in kwargs:
        ax_.set_xlabel(kwargs.get('xlabel'))

def plot_bin_edges(bin_edges: NDArray) -> None:
    for edge in bin_edges:
        ax.axvline(edge, ls='--', alpha=0.5, color='grey')
    ax.axvline(edge, ls='--', alpha=0.5, color='grey', label='bin edges')  # do the last again for the legend

def plot_weights(ax_: plt.Axes, hist_axes: bh.axis, weights_: NDArray, 
                 normalize: bool = False, **kwargs) -> None:
    hist_ = bh.Histogram(hist_axes)
    hist_.fill(weights_)
    plot_hist(ax_, hist_, color='blue', normalize=normalize, **kwargs)
    ax_.set_ylim(0, None)
    ax_.set_xlabel('Weight')
    ax_.set_ylabel('#')






dataframefile: Filename = 'data/new_data_pre.pkl'
histfile: Filename = 'data/hist_weight.pkl'
df = pd.read_pickle(dataframefile)
pt = df.pt
eta = df.eta
real: Mask = df.real
weights_fake = np.load('data/weights_fake.npy')
weights_real = np.load('data/weights_real.npy')


def get_bins_from_length(start: float, stop: float, length: float) -> Tuple[int, float, float]:
    count: int = int((stop-start)/length)
    return (count, start, stop)

pt_validation_bins = get_bins_from_length(15, 275, 1)
eta_validation_bins = (60, -3, 3)

with open(histfile, "rb") as f:
    hist: bh.Histogram = pickle.load(f)

pt_bins: NDArray = hist.axes[0].edges
eta_bins: NDArray = hist.axes[1].edges





############################################################################################################
fig, ax = plt.subplots(figsize=(14, 10))
compare_1d(ax, bh.axis.Regular(*pt_validation_bins), pt, weights_fake, real, which='fake', exclude=(pt > 250))
for edge in pt_bins:
    ax.axvline(edge, ls='--', alpha=0.5, color='grey')
ax.axvline(edge, ls='--', alpha=0.5, color='grey', label='bin edges')  # do the last again for the legend
ax.legend(loc='upper right')
ax.set_xlabel('$p_T$ [GeV]')
plt.tight_layout()
fig.savefig('plots/reweighting_fake_pt.png')


fig, ax = plt.subplots(figsize=(14, 10))
compare_1d(ax, bh.axis.Regular(*eta_validation_bins), eta, weights_fake, real, which='fake', exclude=(pt > 250))
for edge in eta_bins:
    ax.axvline(edge, ls='--', alpha=0.5, color='grey')
ax.axvline(edge, ls='--', alpha=0.5, color='grey', label='bin edges')  # do the last again for the legend
ax.legend(loc='lower center')
ax.set_xlabel('$\eta$')
plt.tight_layout()
fig.savefig('plots/reweighting_fake_eta.png')

############################################################################################################
fig, ax = plt.subplots(figsize=(14, 12))
compare_1d(ax, bh.axis.Regular(*pt_validation_bins), pt, weights_real, real, which='real', exclude=(pt > 250))
plot_bin_edges(pt_bins)
ax.legend(loc='upper right')
ax.set_xlabel('$p_T$ [GeV]')
plt.tight_layout()
fig.savefig('plots/reweighting_real_pt.png')

ax.set_yscale('linear')
ax.set_ylim(0, 0.04)
fig.savefig('plots/reweighting_real_pt_linear.png')



fig, ax = plt.subplots(figsize=(14, 12))
compare_1d(ax, bh.axis.Regular(*eta_validation_bins), eta, weights_real, real, which='real', exclude=(pt > 250))
plot_bin_edges(eta_bins)
ax.legend(loc='lower center')
ax.set_xlabel('$\eta$')
plt.tight_layout()
fig.savefig('plots/reweighting_real_eta.png')

ax.set_yscale('linear')
fig.savefig('plots/reweighting_real_eta_linear.png')

############################################################################################################
weights_real_bins = (20, 0, 0.3)
weights_fake_bins = (25, -1, 85)
#
fig, ax = plt.subplots(figsize=(14, 12))
plot_weights(ax, bh.axis.Regular(*weights_fake_bins), weights_fake, label='fake weights')
ax.legend(loc='upper right')
plt.tight_layout()
fig.savefig('plots/reweighting_fake_weights.png')


fig, ax = plt.subplots(figsize=(14, 12))
plot_weights(ax, bh.axis.Regular(*weights_real_bins), weights_real, label='real weights')
ax.legend(loc='upper right')
plt.tight_layout()
fig.savefig('plots/reweighting_real_weights.png')







print('FINISHED')

plt.show()


