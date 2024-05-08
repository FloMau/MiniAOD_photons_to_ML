import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh

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
eveto = df['eveto']

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
total_mask = pt & eta & shower_shape & one_of & eveto

# rename come cuts for readability
real = df['is_real']==1
fake = ~real
no_eveto = pt & eta & shower_shape & one_of
########################################################################################################################
def compare_cuts(data, binning, cut1, cut2, cut3, cut4, titles, xlabel='', ylabel='', yscale='linear'):
    '''plots 4 subplots which compare real/fake with given cuts applied, and labels them with titles
    cut1-4: masks
    title: iterable of strings'''
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs.flatten()

    if cut1 is None:
        cut1 = np.ones_like(data, dtype=bool)

    ax1.set_title(titles[0])
    ax2.set_title(titles[1])
    ax3.set_title(titles[2])
    ax4.set_title(titles[3])

    histplot(ax1, data[cut1 & real], binning, color='blue', label='real')
    histplot(ax1, data[cut1 & fake], binning, color='red', label='fake')

    histplot(ax2, data[cut2 & real], binning, color='blue', label='real')
    histplot(ax2, data[cut2 & fake], binning, color='red', label='fake')

    histplot(ax3, data[cut3 & real], binning, color='blue', label='real')
    histplot(ax3, data[cut3 & fake], binning, color='red', label='fake')

    histplot(ax4, data[cut4 & real], binning, color='blue', label='real')
    histplot(ax4, data[cut4 & fake], binning, color='red', label='fake')

    for ax in axs.flatten():
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
    return fig


titles = ['no cuts', '$e^-$-veto only', 'other cuts', 'other + $e^-$-veto']
fig1 = compare_cuts(df['pt'], (35, 0, 350), None, eveto, no_eveto, total_mask, titles, xlabel=r'$p_t$ [GeV]', yscale='linear')
fig2 = compare_cuts(df['pt'], (35, 0, 350), None, eveto, no_eveto, total_mask, titles, xlabel=r'$p_t$ [GeV]', yscale='log')
fig3 = compare_cuts(df['eta'], (30, -3, 3), None, eveto, no_eveto, total_mask, titles, xlabel=r'$\eta$', yscale='linear')
fig4 = compare_cuts(df['phi'], (34, -3.2, 3.2), None, eveto, no_eveto, total_mask, titles, xlabel=r'$\varphi$', yscale='linear')

fig1.savefig('plots/eveto_comp_pt1.png')
fig2.savefig('plots/eveto_comp_pt2.png')
fig3.savefig('plots/eveto_comp_eta.png')
fig4.savefig('plots/eveto_comp_phi.png')



plt.show()

