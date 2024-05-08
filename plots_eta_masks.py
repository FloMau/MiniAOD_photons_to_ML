import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh
from get_cuts import get_cutmask

plt.rcParams['font.size'] = 18.0
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.


def get_detectormask(dataframe):
    '''split data into Barrel and Endcap'''
    mask = dataframe['det_ID'] == 1
    return mask

def real_fake(df):
    '''returns the real and fake lines of the df'''
    reals = df[df['is_real'] == 1]
    fakes = df[df['is_real'] == 0]
    return reals, fakes

def histplot(ax, data, binning, **kwargs):
    '''put data in a boosthist with binning and plot it along the giving axis, all kwargs are passed to ax.plot
    returns the histogram'''
    hist = bh.Histogram(bh.axis.Regular(*binning))
    hist.fill(data)
    ax.plot(hist.axes[0].edges[:-1], hist.view() / hist.sum(), ds='steps-post', **kwargs)
    return hist


df = pd.read_pickle('data/combined.pkl')

### cuts
pt = df['pt'] > 25  # no leading photon, because I do single photon studies
transition = (1.44 < df['eta'].abs()) & (df['eta'].abs() < 1.57)
eta = (df['eta'].abs() < 2.5) & (~transition)
HoE = df['HoE'] < 0.08
iso_gamma = df['I_gamma'] < 4.0
iso_track = df['I_tr'] < 6.0

# barrel
barrel = df['det_ID'] == 1
R9_small_barrel = df['r9'] > 0.5
R9_large_barrel = df['r9'] > 0.85
sigma_barrel = df['sigma_ieie'] < 0.015
barrel1 = barrel & R9_small_barrel & sigma_barrel & iso_gamma & iso_track
barrel2 = barrel & R9_large_barrel
barrel = barrel1 | barrel2

# endcap
endcap = df['det_ID'] == 2
R9_small_endcap = df['r9'] > 0.80
R9_large_endcap = df['r9'] > 0.90
sigma_endcap = df['sigma_ieie'] < 0.035
endcap1 = endcap & R9_small_endcap & sigma_endcap & iso_gamma & iso_track
endcap2 = endcap & R9_large_endcap
endcap = endcap1 | endcap2

# combine Masks
shower_shape = HoE & (barrel | endcap)
one_of = (df['r9'] > 0.8) | ((df['I_ch'] / df['pt']) < 0.3) | (df['I_ch'] < 20)
total_mask = pt & eta & shower_shape & one_of

# rename come cuts for readability
pt_cut = pt
eta_cut = eta
real = df['is_real']==1
fake = ~real
large_R9  =(barrel2 | endcap2)
small_R9 = (R9_small_barrel & barrel) | (R9_small_endcap & endcap)
sigma = (sigma_barrel & barrel) | (sigma_endcap & endcap)
########################################################################################################################
def compare_cuts(cut1, cut2, cut3, cut4, titles):
    '''plots 4 subplots with given cuts applied, and labels them with titles
    cut1-4: masks
    title: iterable of strings'''
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs.flatten()

    if cut1 is None:
        cut1 = np.ones_like(eta_cut, dtype=bool)

    ax1.set_title(titles[0])
    ax2.set_title(titles[1])
    ax3.set_title(titles[2])
    ax4.set_title(titles[3])

    histplot(ax1, df['eta'][cut1 & eta_cut & real], (30, -3, 3), color='blue', label='real')
    histplot(ax1, df['eta'][cut1 & eta_cut & fake], (30, -3, 3), color='red', label='fake')

    histplot(ax2, df['eta'][cut2 & eta_cut & real], (30, -3, 3), color='blue', label='real')
    histplot(ax2, df['eta'][cut2 & eta_cut & fake], (30, -3, 3), color='red', label='fake')

    histplot(ax3, df['eta'][cut3 & eta_cut & real], (30, -3, 3), color='blue', label='real')
    histplot(ax3, df['eta'][cut3 & eta_cut & fake], (30, -3, 3), color='red', label='fake')

    histplot(ax4, df['eta'][cut4 & eta_cut & real], (30, -3, 3), color='blue', label='real')
    histplot(ax4, df['eta'][cut4 & eta_cut & fake], (30, -3, 3), color='red', label='fake')

    for ax in axs.flatten():
        ax.axvline( 1.5, color='grey', alpha=0.5)  # transition
        ax.axvline(-1.5, color='grey', alpha=0.5)
        ax.set_xlabel('$\eta$')
        ax.legend()
    return fig

eveto = df['eveto']
titles = ['no cuts', 'eveto', 'all_cuts', 'shower shape cuts']
fig1 = compare_cuts(None, eveto, total_mask, shower_shape, titles)

titles = ['one of', '$H/E$', '$R_9$ & Isolations', 'large $R_9$']
fig2 = compare_cuts(one_of, HoE, (barrel | endcap), large_R9, titles)

titles = ['small $R_9$', '$\sigma_{i\eta i\eta}$', '$I_{tr}$', '$I_{ph}$']
fig3 = compare_cuts(small_R9, sigma, iso_track, iso_gamma, titles)


# fig1.savefig('plots/eta_cuts1.pdf')
# fig2.savefig('plots/eta_cuts2.pdf')
# fig3.savefig('plots/eta_cuts3.pdf')
#
# fig1.savefig('plots/eta_cuts1.png')
# fig2.savefig('plots/eta_cuts2.png')
# fig3.savefig('plots/eta_cuts3.png')





plt.show()

