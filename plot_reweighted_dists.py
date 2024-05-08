import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh
import pickle

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
import myplotparams

# my functions
from mynetworks import load_and_prepare_data
from mymodules import histplot
from mymodules import plot_training
from mymodules import get_centered_binning
from mymodules import plot_dist
from mymodules import plot_output

# my type aliases
Mask = NDArray[bool]


############################################################
datafile: str = 'data/all_data.pkl'
# # for running on laptop:
# datafile = '~/master/all_data.pkl'

df_all: pd.DataFrame = pd.read_pickle(datafile)
df: pd.DataFrame = df_all[df_all.preselection]
real: Mask = df.real
fake: Mask = ~real
pt: pd.Series = df.pt
eta: pd.Series = df.eta
realfake_ratio: float = real.sum()/fake.sum()
weights: NDArray = df.weight.to_numpy()


pt_bins: Tuple[int, float, float] = (25, 0., 250.)
hist_real: bh.Histogram = bh.Histogram(bh.axis.Regular(*pt_bins, underflow=True, overflow=True))
hist_real.fill(pt[real], weight=weights[real])
hist_fake = bh.Histogram(bh.axis.Regular(*pt_bins, underflow=True, overflow=True))
hist_fake.fill(*pt[fake], weight=weights[fake])

fig, axs = plt.subplots(1,2, figsize=(8,10))
ax1, ax2 = axs
histplot(ax1, pt[real], pt_bins, color='blue', linestyle='-')
histplot(ax1, pt[fake], pt_bins, weights=weight_hist.rebin(rebinning).view().sum(axis=0), xlabel='$pt [\mathrm[GeV]]$', color='red')

fig.savefig('plots/pt_reweighted.png')
# plt.savefig('plots/eta_reweighted.png')


plt.show()




