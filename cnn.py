print('starting python')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import argparse

print('general imports done')
import myplotparams

# my functions
from mynetworks import load_and_prepare_data
from mynetworks import rescale

from mymodules import plot_training
from mymodules import plot_output

from myparameters import Parameters
from myparameters import data_from_params, weights_from_params
from myparameters import build_cnn_from_params
print('my imports done')

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask
from mytypes import Callback, Layer
print('typing imports done')


# shorthands
models = keras.models
layers = keras.layers

##################################################################
### data loading and preparing
print('starting proper code:')
parser = argparse.ArgumentParser(description='train vit with hyperparameters from parameterfile')
parser.add_argument('parameterfile', help='file to be read')
paramfile = parser.parse_args().parameterfile

print(paramfile)
params = Parameters(load=paramfile)

### load and prepare the data
df: pd.DataFrame = pd.read_pickle(params['dataframefile'])

# use only barrel
selection: Mask = np.ones(len(df), dtype=bool)
is_barrel: Mask = df['detID'].to_numpy()
if params['barrel_only']:
    selection = is_barrel

pt = df.pt.to_numpy()[selection]
eta = df.eta.to_numpy()[selection]
pileup = df.pileup.to_numpy()[selection]

weights = weights_from_params(params, selection=selection)
weights_test = weights_from_params(params, test_set=True, selection=selection)
(x_train, y_train), (x_test, y_test) = data_from_params(params, selection=selection)


# rescale other input variables
pt_train, pt_test = rescale(params, pt, weights)
eta_train, eta_test = rescale(params, eta, weights)
pileup_train, pileup_test = rescale(params, pileup, weights)

training_inputs = [x_train, pt_train, eta_train, pileup_train]
test_inputs = [x_test, pt_test, eta_test, pileup_test]

###########################################################################
# TODO
input_shape: Tuple[int, int] = x_train[0].shape

### Callbacks
checkpointfile = f'{modeldir}{modelname}_checkpoints.keras'
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', verbose=2, restore_best_weights=True)
checkpointing = keras.callbacks.ModelCheckpoint(checkpointfile, monitor='val_accuracy', save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=4)

# callbacks = [earlystopping, checkpointing, reduce_lr]
# callbacks = [earlystopping, reduce_lr]
callbacks = [earlystopping]


########################################################################
### Callbacks
callbacks: List[Callback] = []
if params['use_checkpointing']:
    checkpointfile = params['modeldir'] + params['modelname'] + '_checkpoints'
    callbacks += [keras.callbacks.ModelCheckpoint(checkpointfile, **params['checkpointing'])]
if params['use_earlystopping']:
    callbacks += [keras.callbacks.EarlyStopping(**params['earlystopping'])]
if params['use_reduce_lr']:
    callbacks += [keras.callbacks.ReduceLROnPlateau(**params['reduce_lr'])]

###########################################################################
### build and train the network
def build_cnn(image_size: int) -> models.Model:
    input_layer = layers.Input(shape=(image_size, image_size, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(64,(3, 3), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(input_layer, output_layer)


model = build_cnn_from_params(params)
model.summary()
print()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              weighted_metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.25,
                    epochs=500,
                    batch_size=512,
                    verbose=2,
                    # class_weight={0: real_to_fake_ratio, 1: 1}
                    sample_weight=weights,
                    callbacks=callbacks
                    )

### save model and history
modelsavefile = modeldir + modelname + '.keras'
historyfile = modeldir + modelname + '_history.npy'
model.save(modelsavefile)
np.save(historyfile, history.history)
print('INFO: model saved as', modelsavefile)
print('INFO: history saved as', historyfile)

############################################################################
### evaluate and print test accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=0)
print('test_accuracy =', test_acc)

### plot training curves
figname = modeldir + modelname + '_training.png'
plot_training(history.history, test_acc, savename=figname)

##############################################################################
### calculate and plot output
y_pred: NDArray = model.predict(x_test, verbose=2).flatten()  # output is shape (..., 1)
real: Mask = y_test.astype(bool)
binning: Tuple[int, float, float] = (int(1/params['output_binwidth']), 0., 1.)

fig, ax = plt.subplots(figsize=(10, 8))
plot_output(ax, y_pred, real, binning)  # real and fake outputs are normalized separately
ax.set_title('Output CNN')
plt.tight_layout()
figname: str = modeldir + modelname + '_output.png'
fig.savefig(figname)
print(f'INFO: model output saved as: {figname}')


print('FINISHED')




