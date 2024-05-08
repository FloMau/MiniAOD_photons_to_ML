import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import argparse

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Mask, Filename

import myplotparams

# my functions
from mynetworks import load_and_prepare_data
from mynetworks import Patches, PatchEncoder, resize_images
from mynetworks import obtain_predictions
from myparameters import Parameters
from myparameters import data_from_params, weights_from_params


# shorthands
layers = keras.layers
models = keras.models
tf.get_logger().setLevel('ERROR')


def process_parser() -> Parameters:
    parser = argparse.ArgumentParser(description='evaluate model on testset')
    parser.add_argument('parameterfile', help='parameterfile to be used')

    args = parser.parse_args()
    return Parameters(load=args.parameterfile)


params = process_parser()
modelname: Filename = params['modeldir'] + params['modelname'] + '.keras'
# model: models.Model = keras.models.load_model(modelname, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
_, (x_train, y_train) = data_from_params(params)
df: pd.DataFrame = pd.read_pickle(params['dataframefile'])
weights = weights_from_params(params, df)

y_pred = obtain_predictions(modelname, rechits=x_train, verbose=params['verbose'])

savename: Filename = params['modeldir'] + params['modelname'] + '_pred.npy'
np.save(savename, y_pred)

print(f'INFO: prediction saves as {savename}')
print('FINISHED')
