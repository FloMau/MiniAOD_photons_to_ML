import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh

import myplotparams
from mymodules import histplot
from mymodules import plot_dist

def hist_and_plot(df, quantity, binning=(100, 0, 1000), logscale='', label=''):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    barrel_mask = df.detID
    for i, ax in enumerate(axs.flat):
        if i==0: data = df[barrel_mask]
        if i==1: data = df[~barrel_mask]
        if i==2: data = df[preselection][barrel_mask]
        if i==3: data = df[preselection][~barrel_mask]

        real, fake = df[df.real], df[~df.real]

        hist1 = bh.Histogram(bh.axis.Regular(*binning))
        hist1.fill(real[quantity])
        ax.plot(hist1.axes[0].edges[:-1], hist1.view()/hist1.sum(), ds='steps-post', color='blue', label='real')

        hist2 = bh.Histogram(bh.axis.Regular(*binning))
        hist2.fill(fake[quantity])
        ax.plot(hist2.axes[0].edges[:-1], hist2.view()/hist2.sum(), ds='steps-post', color='red', label='fake')

        # hist3 = hist1 + hist2
        # ax.plot(hist3.axes[0].edges[:-1], hist3.view()/hist3.sum(), ds='steps-post', color='black', label='total')

        ax.legend()
        if 'x' in logscale:
            ax.set_xscale('log')
        if 'y' in logscale:
            ax.set_yscale('log')
        ax.set_xlabel(quantity)
        if label:
            ax.set_xlabel(label)
            fig.suptitle(label)
        if i==0: ax.set_title('Barrel')
        if i==0: ax.set_ylabel('without cuts')
        if i==1: ax.set_title('Endcap')
        if i==2: ax.set_ylabel('with cuts')

    # fig.tight_layout()
    fig.savefig(f'plots/new_{quantity}.png')
    return fig, ax

def add_vlines_to_current_fig(values, labels=None):
    fig = plt.gcf()
    axs = plt.gca()
    for ax in axs:
        plt.axvline


df = pd.read_pickle('data/new_data.pkl')
preselection = np.load('data/new_preselection.npy') & df.eveto
# preselection = None

plot_dist(df, df['pt'], mask=preselection, binning=(35, 0, 350), yscale='log', xlabel=r'$p_t$ [GeV]', savename='plots/new_pt.png')

plot_dist(df, df['et'], (35, 0, 350), mask=preselection, xlabel=r'$E_t$', savename='plots/new_et.png')
plot_dist(df, df['eta'], (30, -3, 3), mask=preselection,  yscale='linear', xlabel=r'$\eta$', savename='plots/new_eta.png')
plot_dist(df, df['phi'], (30, -3.2, 3.2), mask=preselection,  yscale='linear', xlabel=r'$\varphi$', savename='plots/new_phi.png')

plot_dist(df, df['r9'], (30, 0.2, 1.11), yscale='linear', xlabel=r'$R_9$', savename='plots/new_rt.png')
plot_dist(df, df['HoE'], (30, 0, 0.8), xlabel=r'$H/E$', savename='plots/new_hoe.png')
plot_dist(df, df['sigma_ieie'], (30, 0, 0.05), yscale='linear', xlabel=r'$\sigma_{i\eta i\eta}$', savename='plots/new_sigma_ieie.png')
plot_dist(df, df['ecalIso'], (50, -5, 200), xlabel=r'$ecalPFClusterIso$', savename='plots/new_ecaliso.png')
plot_dist(df, df['hcalIso'], (50, -5, 150), xlabel=r'$hcalPFClusterIso$', savename='plots/new_hcaliso.png')
plot_dist(df, df['I_ch'], (30, 0, 25), xlabel=r'$I_{ch}$', savename='plots/new_ich.png')
plot_dist(df, df['I_gamma'], (30, 0, 25), xlabel=r'$I_{\gamma}$')
plot_dist(df, df['I_n'], (30, 0, 25), xlabel=r'$I_{n}$')
plot_dist(df, df['I_tr'], (30, 0, 10), xlabel=r'$I_{track}$')


plt.show()

