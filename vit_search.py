print('starting python')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse

print('general imports done')
import myplotparams

# my functions
from mynetworks import Patches, PatchEncoder, mlp
from mynetworks import plot_patches, resize_images

from mymodules import plot_training
from mymodules import plot_output

from myparameters import Parameters
from myparameters import data_from_params, weights_from_params
from myparameters import build_vit_from_params
from myparameters import rescale_multiple, split_data
print('my imports done')

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask
from mytypes import Callback, Layer
print('typing imports done')


# shorthands
models = keras.models
layers = keras.layers

print('starting proper code:')
parser = argparse.ArgumentParser(description='train vit with hyperparameters from parameterfile')
parser.add_argument('parameterfile', help='file to be read')
paramfile = parser.parse_args().parameterfile
print(paramfile)
params = Parameters(load=paramfile)
print('Parameters:')
print(params)

### load and prepare the data
df: pd.DataFrame = pd.read_pickle(params['dataframefile'])
other_inputs = [df[key].to_numpy() for key in params['other_inputs']]
pt = df.pt.to_numpy()
eta = df.eta.to_numpy()
rho = df.rho.to_numpy()
HoE = df.HoE.to_numpy()
trackIso = df.I_tr.to_numpy()
hcalIso = df.hcalIso.to_numpy()
converted = df.converted.to_numpy(dtype=int)
convertedOneLeg = df.convertedOneLeg.to_numpy(dtype=int)


weights = weights_from_params(params, selection=None)
weights_test = weights_from_params(params, test_set=True, selection=None)
(x_train, y_train), (x_test, y_test) = data_from_params(params, selection=None)
x_train = resize_images(x_train)
x_test = resize_images(x_test)


needs_scaling = [np.log(pt), eta, rho, HoE, trackIso, hcalIso]
# rescale other input variables
scaled_inputs_train, scaled_inputs_test = rescale_multiple(params, needs_scaling, weights)

converted_train, converted_test = split_data(params, converted)
convertedOneLeg_train, convertedOneLeg_test = split_data(params, convertedOneLeg)
other_train_inputs = scaled_inputs_train + [converted_train, convertedOneLeg_train]
other_test_inputs = scaled_inputs_test + [converted_test, convertedOneLeg_test]
other_train_inputs = np.column_stack(other_train_inputs)
other_test_inputs = np.column_stack(other_test_inputs)

# TODO make plot of rescaled pt eta

# Plot patches of one image
image: NDArray = x_train[np.random.choice(range(x_train.shape[0]))]
plot_patches(image, params['image_size'], params['patch_size'], "image.png")

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

# todo use on_epoch_end for custom (correct) validation loss
#from tensorflow.keras import backend as K
#class printlearningrate(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        optimizer = self.model.optimizer
#        lr = K.eval(optimizer.lr)
#        Epoch_count = epoch + 1
#        print('\n', "Epoch:", Epoch_count, ', LR: {:.2f}'.format(lr))


########################################################################
model = build_vit_from_params(params)
optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              weighted_metrics=['accuracy'])

model.summary()

# todo make fit_params a dict in Parameters
history = model.fit([x_train, other_train_inputs], y_train,
                    sample_weight=weights,
                    callbacks=callbacks,
                    **params['fit_params']
                    )

### save model and history
modelsavefile = params['modeldir'] + params['modelname'] + '.keras'
historyfile = params['modeldir'] + params['modelname'] + '_history.npy'
model.save(modelsavefile)
np.save(historyfile, history.history)
print('model saved as', modelsavefile)
print('history saved as', historyfile)



##################################################################
test_loss, test_acc = model.evaluate([x_test, other_test_inputs],  y_test, sample_weight=weights_test, verbose=params['fit_params']['verbose'])
print('test_accuracy =', test_acc)

### plot training curves
figname: str = params['modeldir'] + params['modelname'] + '_training.png'
plot_training(history.history, test_acc, savename=figname)  # info printed inside function

##############################################################################
### calculate output
y_pred: NDArray = model.predict([x_test, other_test_inputs], verbose=params['fit_params']['verbose']).flatten()  # output is shape (..., 1)
savename: Filename = params['modeldir'] + params['modelname'] + '_pred.npy'
np.save(savename, y_pred)
print(f'INFO: prediction saves as {savename}')


### plot output
real: Mask = y_test.astype(bool)
binning: Tuple[int, float, float] = (int(1/params['output_binwidth']), 0., 1.)

fig, ax = plt.subplots(figsize=(10, 8))
plot_output(ax, y_pred, real, binning)  # real and fake outputs are normalized separately
ax.set_title(f'Output {params["ModelName"]}')
plt.tight_layout()

outputfile: Filename = params['modeldir'] + params['modelname'] + '_output.png'
fig.savefig(outputfile)
print(f'INFO: output saved as {outputfile}')
##############################################################################
### save parameters used
paramfile: Filename = params['modelname'] + '_params.txt'
params.save(paramfile)  # infoprint in save function



print('FINISHED')
