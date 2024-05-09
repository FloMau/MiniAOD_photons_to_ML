from numpy.typing import NDArray
from typing import TypeVar
from numpy import bool_
from keras.callbacks import Callback
from keras.layers import Layer

# my type aliases
Filename = str
Mask = NDArray[bool_]

Particle = TypeVar('Particle')

Callback = Callback
Layer = Layer
