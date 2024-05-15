import sys
from numpy.typing import NDArray
from typing import TypeVar
from numpy import bool_

# my type aliases
Filename = str
Mask = NDArray[bool]
Dict_keys = type({}.keys())

Particle = TypeVar('Particle')

# make sure not to import keras/tensorflow if script has not done so already
if 'keras' in sys.modules:
    from keras.callbacks import Callback
    from keras.layers import Layer
    Callback = Callback
    Layer = Layer
