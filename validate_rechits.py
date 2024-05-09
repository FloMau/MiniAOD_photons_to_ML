import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import boost_histogram as bh
import functools
import operator
import pickle
import myplotparams

import matplotlib.colors as colors
import copy

from typing import Optional, Tuple, Union
from mytypes import Mask, NDArray, Filename


def plot_image(image: NDArray, title: Optional[str] = None, savename: Optional[Filename] = None):
    cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    cmap.set_under('w')

    image[image<1e-6]=1e-6

    plt.figure(figsize=(10, 8))
    im = plt.imshow(image, norm=colors.LogNorm(vmin=1e-6, vmax=image.max()), cmap=cmap, interpolation=None)
    plt.colorbar(im, label='Energy deposition [GeV]')
    plt.xlabel("iphi")
    plt.ylabel("ieta")
    # plt.axvline(10, color='red')
    # plt.axvline(90, color='red')
    # plt.axhline(30, color='red')
    # plt.axhline(70, color='red')
    plt.title(title)
    if savename is not None:
        plt.savefig(savename)
        print(f'INFO: figure saved as {savename}')
    # plt.clf()

def plot_average(data: NDArray, title: Optional[str] = None, savename: Optional[Filename] = None) -> None:
    average = np.mean(data, axis=0)
    plot_image(average, title=title, savename=savename)

def resize_images(array) -> NDArray:
    shape = array.shape
    new_shape = (*shape, 40)
    new = np.ones(new_shape)
    for i in range(40):
        new[:, i] *= array
    return new


print('loading data:')
dataframefile: Filename = 'data/new_data_pre_barrel.pkl'
rechitsfile: Filename = 'data/new_rechits_pre_barrel.npy'
# dataframefile: Filename = 'data/test.pkl'
# rechitsfile: Filename = 'data/test.npy'

df: pd.DataFrame = pd.read_pickle(dataframefile)
rechits: NDArray = np.load(rechitsfile)
real: Mask = df.real.to_numpy()
fake = ~real
eta = df.eta.to_numpy()

idxs = np.indices((11,11))
outer = (idxs == 0) | (idxs==10)
outer = outer[0] | outer[1]
print(outer.shape)

empty = rechits.min()
threshold1 = resize_images(df.pt) * 0.01
threshold2 = resize_images(df.pt) * 0.05
threshold3 = resize_images(df.pt) * 0.1
nonempty_edge = rechits[:,outer]>=empty
small_edge = rechits[:,outer]>=threshold1
middle_edge = rechits[:,outer]>=threshold2
large_edge = rechits[:,outer]>=threshold3
print(nonempty_edge.sum()/rechits.size)
print(nonempty_edge.any(axis=1).sum())
print()
print(large_edge.sum()/rechits.size)
print(large_edge.any(axis=1).sum())


plt.figure(figsize=(10,8))
plt.hist(nonempty_edge.sum(axis=1), label='> 0 GeV', alpha = 0.6)
plt.hist(small_edge.sum(axis=1), label='>  1% $p_t$', alpha = 0.6)
plt.hist(middle_edge.sum(axis=1), label='>  5% $p_t$', alpha = 0.6)
plt.hist(large_edge.sum(axis=1), label='> 10% $p_t$', alpha = 0.6)
plt.xlabel('# of pixels > threshold')
plt.legend(loc='upper right')
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/non_empty_edges.png')
plt.show()
# plot_average(rechits[real & barrel], title='average real in Barrel', savename='plots/rechits_average_barrel_real_large.png')
# plot_average(rechits[fake & barrel], title='average fake in Barrel', savename='plots/rechits_average_barrel_fake_large.png')
# plot_average(rechits[real & endcap], title='average real in endcap', savename='plots/rechits_average_endcap_real_large.png')
# plot_average(rechits[fake & endcap], title='average fake in endcap', savename='plots/rechits_average_endcap_fake_large.png')

print('FINISHED')

