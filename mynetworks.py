import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import tensorflow as tf
import keras

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask, Layer, Callback

models = keras.models
layers = keras.layers


def set_seeds(seed: int) -> None:
    import random
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def load_data(datafile: Filename, rechitfile: Filename, preselectionfile: Optional[Filename]=None,
              selection: Optional[Mask] = None
              ) -> Tuple[NDArray, Mask]:
    df: pd.DataFrame = pd.read_pickle(datafile)
    if preselectionfile is not None:
        preselection: Mask = np.load(preselectionfile) * df.eveto
    else:
        preselection: Mask = np.ones(len(df), dtype=bool)
    if selection is not None:
        preselection *= selection
    rechits: NDArray = np.load(rechitfile)[preselection]
    real: Mask = df['real'][preselection]
    # print('preselectionfile =', preselectionfile, end='\n')
    return rechits, real

def prepare_data(rechits: NDArray, real: Mask, test_fraction: float, use_log: bool = False
                 ) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    if use_log:
        rechits[rechits<1e-6] = 1e-6
        rechits = np.log(rechits)

    split_idx: int = int(len(real)*(1-test_fraction))
    train_hits, train_labels = rechits[:split_idx], real[:split_idx]
    test_hits, test_labels = rechits[split_idx:], real[split_idx:]

    x_train = train_hits
    x_test = test_hits
    y_train = train_labels.astype(int)
    y_test = test_labels.astype(int)
    return (x_train, y_train), (x_test, y_test)

# todo rewrite
def load_and_prepare_data(datafile: Filename, rechitfile: Filename, test_fraction: float, 
                          preselectionfile: Optional[Filename] = None, use_log: bool = False,
                          selection: Optional[Mask] = None) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    rechits, real = load_data(datafile, rechitfile, preselectionfile, selection=selection)
    return prepare_data(rechits, real, test_fraction, use_log=use_log)

def cnn_block(x, filters=16, num_convolutions=2, kernel_size=(3, 3), pool_size=(2,2),
              activation='relu', padding='same', **kwargs):
    """
    simple convolutional building block consisting of convolutional layers followed by Maxpooling
    :param x: inputlayer
    :param filters: int, dimension of outputspace, default is 16
    :param kernel_size: tuple/list of 2 int, size of con2d filters,  default is (3,3)
    :param pool_size: tuple/list of 2 int, maxpooling size,  default is (2,2)
    :param num_convolutions: int, how many convolutional layers, default is 2

    :param kwargs: are passed to ConvD and MaxPool2D
    :return: outputlayer
    """
    for _ in range(num_convolutions):
        x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, **kwargs)(x)
    x = layers.MaxPool2D(pool_size=pool_size, padding=padding)(x)
    return x

def dense_block(x, nodes, activation='relu'):
    """
    simple dense network block
    :param x: inputlayer
    :param nodes: int/iterable of ints, one int for every layer describing the amount of nodes per layer
    :param activation: activation function, default is ReLu
    :return: outputlayer
    """
    try:
        iter(nodes)
    except TypeError:
        nodes = [nodes]

    for num in nodes:
        x = layers.Dense(num, activation='relu')(x)
    return x

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

# def plot_patches(image: NDArray, image_size: int, patch_size: int, outfile: Optional[str] = None) -> None:
#     plt.figure(figsize=(4, 4))
#     plt.imshow(image.astype("uint8"))
#     plt.axis("off")
#     plt.savefig(outfile)

#     patches = Patches(patch_size)(image[tf.newaxis, ...])
#     print(patches.shape)
#     print(f"Image size: {image_size} X {image_size}")
#     print(f"Patch size: {patch_size} X {patch_size}")
#     print(f"Patches per image: {patches.shape[1]}")
#     print(f"Elements per patch: {patches.shape[-1]}")

#     n = int(np.sqrt(patches.shape[1]))
#     plt.figure(figsize=(4, 4))
#     for i, patch in enumerate(patches[0]):
#         ax = plt.subplot(n, n, i + 1)
#         patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
#         plt.imshow(patch_img.numpy().astype("uint8"))
#         plt.axis("off")
#     plt.savefig("patched_" + outfile)

def plot_patches(image: NDArray, image_size: int, patch_size: int, outfile: Optional[str] = None) -> None:
    plt.figure(figsize=(4, 4))
    cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    cmap.set_under('w')
    
    image[image<1e-6]=1e-6
    print(image.shape)
    im = plt.imshow(image[:,:,0], norm=mpl.colors.LogNorm(vmin=1e-6, vmax=image.max()), cmap=cmap, interpolation=None)
    plt.colorbar(im, label='Energy deposition [GeV]')
    # plt.axis("off")
    plt.tight_layout()
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
        plt.imshow(patch_img.numpy()[:,:,0], norm=mpl.colors.LogNorm(vmin=1e-6, vmax=image.max()), cmap=cmap, interpolation=None)
        # plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(outfile.split('.')[0] + '_patched.png')


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
    """adds MLP on top of given layer
    MLP consists of one Dense layer with as many nodes per entry in hiddenunits"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def resize_images(array: NDArray) -> NDArray:
    shape = array.shape
    new_shape = (*shape, 3)
    new = np.ones(new_shape)
    for last_layer in range(3):
        new[:, :, :, last_layer] *= array
    return new

###########################################################################

def obtain_predictions(modelname: Filename, rechits: NDArray, verbose: int = 2) -> NDArray:
    print('\n\n')
    print(f'INFO: loading model: {modelname}')
    model: models.Model = keras.models.load_model(modelname,
                                                    custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder})
    print(model.summary())
    print('\n\n')
    if 'vit' in modelname.lower():
        rechits_3d: NDArray = resize_images(rechits)
        predictions: NDArray = model.predict(rechits_3d, verbose=verbose).flatten()
    else:
        predictions: NDArray = model.predict(rechits, verbose=verbose).flatten()
    return predictions

