import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import argparse
import myplotparams

from mynetworks import Patches, PatchEncoder, resize_images
from numpy.typing import NDArray
from mytypes import Mask

def setup_parser():
    parser = argparse.ArgumentParser(description='plot output of model providing a testset', prog='plot_output_dist.py')
    parser.add_argument('model', help='model to be used')
    parser.add_argument('--labels', default='data/all_data_preselected.pkl', help='file containing the labels of the data')
    parser.add_argument('--rechits', default='data/rechits/all_rechits_preselected.npy', help='file containing the rechits')
    parser.add_argument('--preselectionfile', help='use preselection if given')
    parser.add_argument('--vit', action='store_true', help='resize input to (...,...,3) for Transformer models')
    parser.add_argument('--bdt', action='store_true', help='plot output of CMS BDT')
    # todo think about use_log

    args = parser.parse_args()
    return args

# args = setup_parser()
# modelfile, datafile, rechitfile, preselectionfile = args.model, args.labels, args.rechits, args.preselectionfile
# use_bdt = args.bdt
# # path = '/home/tim/uni/master/master/cluster/'
# print(preselectionfile)

### load the model
# modelname = modelfile.split('.')[0]
# model = keras.models.load_model(modelfile, custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder})

### load and prepare the data
# from mynetworks import load_and_prepare_data
# _, (x_test, y_test) = load_and_prepare_data(datafile, rechitfile, 0.2, preselectionfile=preselectionfile, use_log=True)
# if args.vit:
#     x_test = resize_images(x_test)
# real = y_test.astype(bool)
# real_fraction = real.sum()/real.size
# print(real.size)
# print(real_fraction)
# print(1-real_fraction)
#


### predict scores
# y_pred = model.predict(x_test).flatten()  # output shape is (X, 1)
df: pd.DataFrame = pd.read_pickle('data/new_data_pre_barrel.pkl')
real: Mask = df.real[:int(0.8*len(df))]  # only use test set
bdt: NDArray = df.bdt3.to_numpy()[:int(0.8*len(df))]
print(bdt.shape)
print(len(real))
print((~real).sum())
modelname: str = 'bdt'

from mymodules import get_centered_binning
from mymodules import plot_output
binwidth = 0.05
binning = (int(1/binwidth), 0., 1.)
ax: plt.Axes
fig, ax = plt.subplots(figsize=(10, 8))
plot_output(ax, bdt, real, binning)  # real and fake outputs are normalized separately
ax.set_title('BDT Output')
ax.grid(True)
plt.tight_layout()

figname = f'models/{modelname}_output.png'  # models/ is part of modelname
if figname is not None:
    fig.savefig(figname)
    print('fig saves as:', figname)




plt.show()


