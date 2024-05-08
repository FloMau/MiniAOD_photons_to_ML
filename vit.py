print('starting python')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
print('general imports done')

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask
print('typing imports done')

import myplotparams

# my functions
from mynetworks import load_and_prepare_data
from mymodules import plot_training
from mymodules import get_centered_binning
from mymodules import plot_output
print('my imports done')



# shorthands
models = keras.models
layers = keras.layers

Callback = keras.callbacks.Callback
Layer = keras.layers.Layer
print('starting proper code:')
###########################################################################
class Patches(layers.Layer):
    def __init__(self, patch_size: int, **kwargs) -> None:
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images: NDArray) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

def plot_patches(image: NDArray, image_size: int, patch_size: int, outfile: Optional[str] = None) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    plt.savefig(outfile)

    patches = Patches(patch_size)(image[tf.newaxis, ...])
    print(patches.shape)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
    plt.savefig("patched_" + outfile)


###########################################################################
### load and prepare data
modeldir: str = 'models/'
assert modeldir[-1] == '/'
modelname: str = 'vit'
dataframefile: Filename = 'data/new_data_pre.pkl'
rechitsfile: Filename = 'data/new_rechits_pre.npy'
# preselectionfile: Filename = 'data/all_preselection.npy'

df: pd.DataFrame = pd.read_pickle(dataframefile)
weights: pd.Series = df.weight[:int(0.8*len(df))]
print(type(weights))

### load and prepare the data
(x_train, y_train), (x_test, y_test) = load_and_prepare_data(dataframefile, rechitsfile,
                                                             test_fraction=0.2,
                                                             preselectionfile=None, use_log=True)

# todo use slice object

def resize_images(array: NDArray) -> NDArray:
    shape = array.shape
    new_shape = (*shape, 3)
    new = np.ones(new_shape)
    for last_layer in range(3):
        new[:, :, :, last_layer] *= array
    # new = np.moveaxis(new, [0, 1, 2, 3], [1, 2, 3, 0])
    return new


x_train = resize_images(x_train)  # reshape to RGB by coping the values 3 times
x_test = resize_images(x_test)




fake_fraction: float = 1 - y_train.sum()/y_train.size
real_to_fake_ratio: float = (1-fake_fraction)/fake_fraction
# input_shape = (1, *x_train.shape)
input_shape: List[int] = [11, 11, 3]
print('\n\n\n')
print('trainshape:', x_train.shape)
print('inputshape', input_shape)
print('\n\n\n')
#####################################################################
### prepare ViT
# Small values to enable fast training -> will not achieve good performance
learning_rate: float = 1e-3
batch_size: int = 1024
num_epochs: int = 500

image_size: int = input_shape[0]
patch_size: int = 4  # Size of the patches to be extract from the input images
num_patches: int = (image_size // patch_size) ** 2
projection_dim: int = 16
num_heads: int = 2
transformer_units: List[int] = [projection_dim]  # Size of the transformer layers
transformer_layers: int = 2
mlp_head_units: List[int] = [32]  # Size of the dense layers of the final classifier

# Plot patches of one image
image: NDArray = x_train[np.random.choice(range(x_train.shape[0]))]
plot_patches(image, image_size, patch_size, "image.png")

########################################################################
### Callbacks
checkpointfile: Filename = f'models/{modelname}_checkpoints'
earlystopping: Callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, mode='auto',
                                                        verbose=2, restore_best_weights=True)
# checkpointing = keras.callbacks.ModelCheckpoint(checkpointfile, monitor='val_accuracy', save_best_only=True)
reduce_lr: Callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, verbose=2)

#from tensorflow.keras import backend as K
#class printlearningrate(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        optimizer = self.model.optimizer
#        lr = K.eval(optimizer.lr)
#        Epoch_count = epoch + 1
#        print('\n', "Epoch:", Epoch_count, ', LR: {:.2f}'.format(lr))

# callbacks: List[Callback] = [earlystopping, checkpointing, reduce_lr]#, printlearningrate]
# callbacks: List[Callback] = [earlystopping, reduce_lr]
callbacks: List[Callback] = [earlystopping]
###########################################################################

# Define PatchEncoder Layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int, **kwargs) -> None:
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


# Define MLP
def mlp(x: Layer, hidden_units: List[int], dropout_rate: float) -> Layer:
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


########################################################################

# Build model
inputs = layers.Input(shape=input_shape)
# Create patches.
patches = Patches(patch_size)(inputs)
# Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

# Create a [batch_size, projection_dim] tensor.
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(representation)
representation = layers.Dropout(0.5)(representation)
# Add MLP.
features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)

# Classify outputs: Simple implementation without separate class token
# logits = layers.Dense(num_classes)(features)
outputs = layers.Dense(1, activation='sigmoid')(features)

# Create the Keras model.
model = keras.Model(inputs=inputs, outputs=outputs)  # outputs was logits in original

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              weighted_metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, validation_split=0.25,  # 0.25 means split of 60/20/20
                    epochs=num_epochs,
                    batch_size=batch_size,
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
print('model saved as', modelsavefile)
print('history saved as', historyfile)

#############################################################################
### evaluate and print testaccuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('test_accuracy =', test_acc)

### plot training curves
figname: str = modeldir + modelname + '_training.png'
plot_training(history.history, test_acc, savename=figname)  # info printed inside function
##############################################################################
### calclate and plot output
y_pred: NDArray = model.predict(x_test, verbose=2).flatten()  # output is shape (..., 1)
real: Mask = y_test.astype(bool)
binning: Tuple[int, float, float] = get_centered_binning(0.1, 0, 1)

fig, ax = plt.subplots(figsize=(10, 8))
plot_output(ax, y_pred, real, binning)  # real and fake outputs are normalized separately
ax.set_title('Output ViT')
plt.tight_layout()
fig.savefig(modeldir + modelname + '_output.png')



print('FINISHED')