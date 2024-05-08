from numpy.typing import NDArray
from typing import TypeVar
import numpy as np
import keras

# my type aliases
Filename = str
Mask = NDArray[np.bool_]

Particle = TypeVar('Particle')

Callback = keras.callbacks.Callback
Layer = keras.layers.Layer
