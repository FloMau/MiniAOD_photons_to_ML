import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow import keras
# import tensorflow as tf
import argparse
import sklearn.metrics as skm

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Mask, Filename

import myplotparams

# my functions
from mynetworks import load_and_prepare_data
from myparameters import Parameters, data_from_params, weights_from_params
from myparameters import check_params_work_together
from myparameters import get_training_slice, get_test_slice

# shorthands
# layers = keras.layers
# models = keras.models
# tf.get_logger().setLevel('ERROR')


def get_tpr(cut: float, predictions: NDArray, true_values: NDArray,) -> float:
    larger: Mask = predictions > cut
    true_positives: Mask = true_values[larger]
    tpr: float = true_positives.sum()/true_values.sum()
    return tpr

def interpolate(x: NDArray, y: NDArray, x_new: NDArray) -> NDArray:
    # x is tpr
    # y is fpr
    return np.interp(x_new, x, y)


def plot_roc(axis: plt.Axes, predictions: NDArray, true_values: NDArray, weights: NDArray,
             threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
    """plots ROC as usual in HEP = true positive rate vs background rejection (1/fpr), xlim will be > threshold
    if threshold is undesiered, set to any negative value or None"""
    if threshold is None:
        threshold: float = -1.
    fpr, tpr, _ = skm.roc_curve(true_values, predictions, sample_weight=weights)
    mask: Mask = tpr > threshold
    axis.plot(tpr[mask], 1/(fpr[mask]), linewidth=2, **kwargs)
    if fifty_percent_line:
        fifty: float = get_tpr(0.5, predictions, true_values)
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)

    axis.set_title('ROC')
    axis.set_xlabel('True positives rate')
    axis.set_ylabel('Background rejection')
    axis.set_xlim(threshold, 1.0)
    # axis.set_ylim(0, 1)
    axis.grid(True)
    axis.grid(True)
    axis.grid(True)
    # axis.set_aspect('equal')
    axis.legend(loc='upper right')
    plt.tight_layout()

def plot_roc_classic(axis: plt.Axes, predictions: NDArray, true_values: NDArray, weight: NDArray,
             threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
    """plots ROC as usual in HEP = true positive rate vs background rejection (1/fpr), xlim will be > threshold
    if threshold is undesiered, set to any negative value or None"""
    if threshold is None:
        threshold: float = -1.
    fpr, tpr, _ = skm.roc_curve(true_values, predictions, sample_weight=weights)
    mask: Mask = tpr > threshold

    axis.plot(tpr[mask], fpr[mask], linewidth=2, **kwargs)
    if fifty_percent_line:
        fifty: float = get_tpr(0.5, predictions, true_values)
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)

    axis.set_title('ROC classic')
    axis.set_xlabel('True positive rate')
    axis.set_ylabel('False positive rate')
    axis.set_xlim(threshold, 1.0)

    axis.grid(True)
    axis.legend(loc='upper left')
    plt.tight_layout()

def plot_roc_ratio(axis: plt.Axes, predictions_base: NDArray, predictions_compare: NDArray, true_values: NDArray,
                   weights: NDArray, 
                   threshold: float = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:

    fpr_base, tpr_base, _ = skm.roc_curve(true_values, predictions_base, sample_weight=weights)
    fpr_comp, tpr_comp, _ = skm.roc_curve(true_values, predictions_compare, sample_weight=weights)
    mask: Mask = tpr_base > threshold

    fpr_interp = np.interp(tpr_base, tpr_comp, fpr_comp)

    rej_base = 1/fpr_base
    rej_interp = 1/fpr_interp

    axis.plot(tpr_base[mask], rej_interp[mask]/rej_base[mask], linewidth=2, **kwargs)

    if fifty_percent_line:
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        fifty: float = get_tpr(0.5, predictions_compare, true_values)
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)
    axis.axhline(1, color='black', alpha=0.8, ls='--', zorder=-1)

    axis.set_title('ROC ratios')
    axis.set_xlabel('True positives rate')
    axis.set_ylabel('Background rejection ratio')
    axis.set_xlim(threshold, 1.)
    ylim = max(np.abs(1-np.array(axis.get_ylim())))
    axis.set_ylim(1-ylim, 1+ylim)

    axis.grid(True)
    axis.legend(loc='upper left')
    plt.tight_layout()


def process_parser() -> Tuple[List[Parameters], Filename]:
    parser = argparse.ArgumentParser(description='plot roc of models ', prog='plot_roc_dist.py')
    parser.add_argument('parameterfilenames', nargs='+', help='model to be used')
    parser.add_argument('--figname', default='models/roc.png')

    args = parser.parse_args()
    param_list_: List[Parameters] = [Parameters(load=file) for file in args.parameterfilenames]
    check_params_work_together(param_list_)

    figname: Filename = args.figname
    return param_list_, figname


### load and prepare the data
param_list, figname = process_parser()
df = pd.read_pickle(param_list[0]['dataframefile'])
weights: NDArray = weights_from_params(param_list[0], test_set=True)

_, (x_test, y_test) = data_from_params(param_list[0])


### get the model predictions
y_pred_list = [np.load(param['modeldir'] + param['modelname'] + '_pred.npy') for param in param_list]
pred_bdt: NDArray = df['bdt3'].to_numpy()
pred_bdt = get_test_slice(param_list[0], pred_bdt)
#TODO filter out the photons with nan pileup

######################################################################################
### plot ROC
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
plot_roc(ax1, pred_bdt, y_test, weights, label='BDT')#, color='orange')
plot_roc_classic(ax3, pred_bdt, y_test, weights, label='BDT')#, color='orange')

for i, param in enumerate(param_list):
    plot_roc(ax1, y_pred_list[i], y_test, weights, label=param['modelname'])
    plot_roc_classic(ax3, y_pred_list[i], y_test, weights, label=param['modelname'])
    plot_roc_ratio(ax2, pred_bdt, y_pred_list[i], y_test, weights, label=f'{param["modelname"]}/BDT')
# ax2.set_title(None)
# ax2.set_ylabel('ratio')
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
# plt.subplots_adjust(hspace=0.0)

if figname is not None: 
    fig2name = f'{figname.split(".")[0]}_ratio.png'
    fig3name = f'{figname.split(".")[0]}_classic.png'
    fig1.savefig(figname)
    fig2.savefig(fig2name)
    fig3.savefig(fig3name)
    print('INFO: fig saves as:', figname, fig2name, fig3name)


print('FINISHED')
plt.show()

