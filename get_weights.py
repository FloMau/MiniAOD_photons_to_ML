import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import boost_histogram as bh
import functools
import operator
import pickle
import argparse
import myplotparams

from typing import Optional, Tuple
from mytypes import Mask, NDArray, Filename

def make_normalized_2dhist(data1, data2, binning1, binning2):
    hist = bh.Histogram(bh.axis.Regular(*binning1), bh.axis.Regular(*binning2))
    hist.fill(*[data1, data2])
    areas = functools.reduce(operator.mul, hist.axes.widths)
    pdd2d = hist.view() / hist.sum() / areas
    return pdd2d, hist

def plot_2dhist(hist):
    plt.pcolormesh(*hist.axes.edges.T, hist.view().T, norm=mpl.colors.LogNorm())


def process_args() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description='get weights and save them')
    parser.add_argument('--datafile', default='data/new_data_pre.pkl', help='data to be used')
    parser.add_argument('--preselectionfile', help='apply preselection from given file')
    args = parser.parse_args()

    datafile_: Filename = args.datafile
    preselectionfile: Optional[Filename] = args.preselectionfile

    df_ = pd.read_pickle(datafile_)
    preselection: Mask = np.ones(len(df_))
    if preselectionfile is not None:
        preselection = np.load(preselectionfile)

    return df_[preselection]

############################################################
df = process_args()

real = df.real
fake = ~real
pt = df.pt
eta = df.eta
ratio = real.sum()/fake.sum()

print('# fakes with pt>150:', ((pt>150) & fake).sum())
print('# fakes with pt>250:', ((pt>250) & fake).sum())
print('# fakes with pt>350:', ((pt>350) & fake).sum())
print(f'real:fake ratio: {real.sum()}:{fake.sum()}={int(ratio)}:1')

### decide bins:
pt_bins = np.array([0, *np.arange(25, 55, step=2.5), 55, 65, 75, 100, 125, 150, 200, 250])

num_merge = 3
eta_bins_barrel = np.linspace(-1.44, 1.44, 4*num_merge+1)  # make barrel bins smaller and merge the ones for high pt
eta_bins_endcap1 = np.linspace(-2.5, -1.57, 2)
eta_bins_endcap2 = np.linspace(1.57, 2.5, 2)
eta_bins = np.append(eta_bins_endcap1, eta_bins_barrel)
eta_bins = np.append(eta_bins, eta_bins_endcap2)  # no doublecounting because of transtion region


# make histograms
# pt has values > last_bin_edge; use overflow bin
hist_real = bh.Histogram(bh.axis.Variable(pt_bins, underflow=True, overflow=True),
                         bh.axis.Variable(eta_bins, underflow=True, overflow=True))
hist_real.fill(*[pt[real], eta[real]])

hist_fake = bh.Histogram(bh.axis.Variable(pt_bins, underflow=True, overflow=True),
                         bh.axis.Variable(eta_bins, underflow=True, overflow=True))
hist_fake.fill(*[pt[fake], eta[fake]])


##############################################################################
## add up last pt eta bins
segment_list = [2, 5, 8, 11, 14]  # end is exclusive when slicing
in_endcap = np.abs(hist_fake[-1, :].axes.centers[0]) > 1.44
last_real = hist_real[-1, :].copy()
last_fake = hist_fake[-1, :].copy()
for i in range(len(in_endcap)):
    #skip endcap
    if in_endcap[i]: continue

    segment_idx = (i-2) // num_merge
    hist_real[-1, i] = last_real[segment_list[segment_idx] : segment_list[segment_idx+1]].sum()
    hist_fake[-1, i] = last_fake[segment_list[segment_idx] : segment_list[segment_idx+1]].sum()

##############################################################################
# plot histograms
fig_real = plt.figure(figsize=(10, 8), constrained_layout=True)
plot_2dhist(hist_real)
plt.title('reals after preselection')
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('$\eta$')
plt.colorbar(label='#')
plt.ylim(-1.5, 1.5)

fig_fake = plt.figure(figsize=(10, 8), constrained_layout=True)
plot_2dhist(hist_fake)
plt.title('fakes after preselection')
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('$\eta$')
plt.colorbar(label='#')
plt.ylim(-1.5, 1.5)


# get weights
pt_idxs = np.digitize(pt, pt_bins)  # digittize checks bin[i-1]<=x<bin[i] -> fits if flow=True
eta_idxs = np.digitize(eta, eta_bins)
weight_hist = hist_real/hist_fake
weights = weight_hist.view(flow=True)[pt_idxs, eta_idxs]


# plot weights
fig_weights_fake = plt.figure(figsize=(10, 8), constrained_layout=True)
plt.pcolormesh(*weight_hist.axes.edges.T, weight_hist.view().T)
plt.title(f'Weights after preselection; real:fake={int(ratio)}')
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('$\eta$')
plt.colorbar(label='weights for fake photons')
plt.ylim(-1.5, 1.5)

# plot weights
fig_weights_real = plt.figure(figsize=(10, 8), constrained_layout=True)
plt.pcolormesh(*weight_hist.axes.edges.T, 1/weight_hist.view().T, norm=mpl.colors.LogNorm())
plt.title(f'Weights after preselection; real:fake={int(ratio)}')
plt.xlabel('$p_t$ [GeV]')
plt.ylabel('$\eta$')
plt.colorbar(label='weights for real photons')
plt.ylim(-1.5, 1.5)
###########################################################################
# save figures
fignames = 'plots/hist_real.png', 'plots/hist_fake.png', 'plots/hist_weights_fake.png', 'plots/hist_weights_real.png'
fig_real.savefig(fignames[0])
fig_fake.savefig(fignames[1])
fig_weights_fake.savefig(fignames[2])
fig_weights_real.savefig(fignames[3])
print(f'INFO: figures saves as {fignames}')

# save hist for possible future use
weight_hist_file = 'data/hist_weight.pkl'
with open(weight_hist_file, "wb") as file:
    pickle.dump(weight_hist, file)
print(f'INFO: weight_hist saved as {weight_hist_file}')

###########################################################################
### save weights
weights_fake = weights
weights_real = 1/weights

weights_fake[real] = 1.
weights_real[fake] = 1.

weights_fake[pt > 250] = 0.
weights_real[pt > 250] = 0.

# TODO add to process args
fake_file = 'data/weights_fake.npy'
real_file = 'data/weights_real.npy'
np.save(fake_file, weights_fake)
print(f'INFO: fake weights saved as {fake_file}')
np.save(real_file, weights_real)
print(f'INFO: real weights saved as {real_file}')


print('FINISHED')

plt.show()


