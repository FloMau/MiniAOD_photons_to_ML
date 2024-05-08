import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import boost_histogram as bh
import functools
import operator

plt.rcParams['font.size'] = 18.0
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.


def histplot(ax, data, binning, normalizer=None, **kwargs):
    """
    put data in a boosthist with binning and plot it along the giving axis
    data is normalized to normalizer: int or summable, is set to 1 if False or set to hist.sum() if None
    kwargs are passed to plot
    returns the histogram
    """
    hist = bh.Histogram(bh.axis.Regular(*binning))
    hist.fill(data)
    if normalizer is None:
        normalizer = hist.sum()
    elif normalizer is False:
        normalizer = 1
    elif isinstance(normalizer, bh.Histogram):
        normalizer = normalizer.sum()
    else:
        normalizer = bh.Histogram(bh.axis.Regular(*binning)).fill(normalizer).sum()

    ax.plot(hist.axes[0].edges[:-1], hist.view() / normalizer, ds='steps-post', **kwargs)
    return hist

def make_2dhist(data1, data2, binning1, binning2):
    hist = bh.Histogram(bh.axis.Regular(*binning1), bh.axis.Regular(*binning2))
    hist.fill(*[data1, data2])
    areas = functools.reduce(operator.mul, hist.axes.widths)
    pdd2d = hist.view() / hist.sum() / areas
    return pdd2d, hist

def plot_2dhist(hist):
    plt.pcolormesh(*hist.axes.edges.T, hist.view().T, norm=mpl.colors.LogNorm())


df = pd.read_pickle('data/combined.pkl')
preselection = np.load('data/preselection.npy')
df = df[preselection & df.eveto]

pt = df['pt']
eta = df['eta']
real = df['real']
fake = ~real

pt_bins = (35, 0, 350)
eta_bins = (30, -3, 3)

plt.figure(figsize=(10, 8), constrained_layout=True)
_, hist = make_2dhist(pt, eta, pt_bins, eta_bins)
plot_2dhist(hist)
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('$\eta$')
plt.colorbar(label='#')
plt.savefig('plots/eta_pt_bins.pdf')
plt.savefig('plots/eta_pt_bins.png')

#################################################################################
num_real = df['num_real']
num_fake = df['num_fake']
num = num_real + num_fake  # total
pt_bins = (35, 0, 350)
num_bins = (num.max()-num.min()+1, np.min(num)-0.5, np.max(num)+0.5)

hist = bh.Histogram(bh.axis.Regular(*pt_bins), bh.axis.Regular(*num_bins))
hist.fill(*[pt[real], num_real[real]])
plt.figure(figsize=(10, 8), constrained_layout=True)
plot_2dhist(hist)
plt.title('real')
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('# of real in event')
plt.colorbar(label='#')
plt.savefig('plots/pt_num_real.pdf')
plt.savefig('plots/pt_num_real.png')

num_bins = (num.max()-num.min()+1, np.min(num)-0.5, np.max(num)+0.5)
hist = bh.Histogram(bh.axis.Regular(*pt_bins), bh.axis.Regular(*num_bins))
hist.fill(*[pt[fake], num_fake[fake]])
plt.figure(figsize=(10, 8), constrained_layout=True)
plot_2dhist(hist)
plt.title('fakes')
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('# of fake in event')
plt.colorbar(label='#')
plt.savefig('plots/pt_num_fake.pdf')
plt.savefig('plots/pt_num_fake.png')





plt.show()

