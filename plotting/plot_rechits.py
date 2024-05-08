import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import boost_histogram as bh
import matplotlib.colors as colors
import copy

import myplotparams


def plot_image(image, title, savename=None):
    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under('w')

    image[image<1e-6]=1e-6

    plt.figure()
    im = plt.imshow(image, norm=colors.LogNorm(vmin=1e-6, vmax=image.max()), cmap=cmap, interpolation=None)
    plt.colorbar(im, label='Energy deposition [GeV]')
    plt.xlabel("iphi")
    plt.ylabel("ieta")
    plt.title(title)
    if savename is not None:
        plt.savefig(savename)
    # plt.clf()

def load_rechits(file='combined.npy'):
    return np.load(file)


df = pd.read_pickle("data/data_barrel_pre.pkl")
rechits = load_rechits('data/rechits_barrel_pre.npy')
# cuts =

pt = df.pt.to_numpy()
real = df.real
fake = ~real

from mynetworks import plot_patches
from mynetworks import resize_images
idx = np.random.randint(len(rechits)-1)
image = rechits[idx:idx+2]  # take multiple to maintain rechit dimensionality for resizing
image = resize_images(image)[0] 
plot_patches(image, 11, 4, outfile='plots/test_patch4.png')
plot_patches(image, 11, 3, outfile='plots/test_patch3.png')



# selection = (pt>200) & real
# idx_large = np.random.choice(np.arange(len(pt))[selection])
# large_pt = pt[idx_large]
# plot_image(image, title=f'real Photon, pt={large_pt: .1f} GeV', savename=f'plots/photon_large_pt_real.png')

# idx_large = np.random.choice(np.arange(len(df.pt))[(df.pt>200) & fake])
# large_pt = pt[idx_large]
# image = rechits[idx_large]
# plot_image(image, title=f'fake Photon, pt={large_pt: .1f} GeV', savename=f'plots/photon_large_pt_fake.png')



# idx_large = np.random.choice(np.arange(len(df.pt))[(df.pt>40) & (df.pt<60) & real])
# large_pt = pt[idx_large]
# image = rechits[idx_large]
# plot_image(image, title=f'fake Photon, pt={large_pt: .1f} GeV', savename=f'plots/photon_real.png')

# idx_large = np.random.choice(np.arange(len(df.pt))[(df.pt>40) & (df.pt<60) & fake])
# large_pt = pt[idx_large]
# image = rechits[idx_large]
# plot_image(image, title=f'fake Photon, pt={large_pt: .1f} GeV', savename=f'plots/photon_fake.png')


# idx_large = np.random.choice(np.arange(len(df.pt))[(df.pt<40) & real])
# large_pt = pt[idx_large]
# image = rechits[idx_large]
# plot_image(image, title=f'fake Photon, pt={large_pt: .1f} GeV', savename=f'plots/photon_small_pt_real.png')

# idx_large = np.random.choice(np.arange(len(df.pt))[(df.pt<40) & fake])
# large_pt = pt[idx_large]
# image = rechits[idx_large]
# plot_image(image, title=f'fake Photon, pt={large_pt: .1f} GeV', savename=f'plots/photon__small_pt_fake.png')


# for i, image in enumerate(x):
    # realfake = 'real' if df.real else 'fake'
    # savedir = 'pics/test/'
    # 
# 
# 
    # plot_image(image, title=f'{i+1:02d} Photon ({realfake})', savename=f'{savedir}{i+1:02d}_Photon_({realfake})')
    # if i>25: break
    # plt.close()

# todo more images
# todo better classification with real/fakes
# todo find out where the empty rechits come from
# todo folien, I guess

plt.show()


