import sys
import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Tuple

# my type aliases
Filename = str
Mask = NDArray[np.bool_]
Sparse = Tuple[NDArray[np.float32], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]

Particle = TypeVar('Particle')

# make sure not to import keras/tensorflow if script has not done so already
if 'keras' in sys.modules:
    from keras.callbacks import Callback
    from keras.layers import Layer
    Callback = Callback
    Layer = Layer
