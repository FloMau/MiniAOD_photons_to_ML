from numpy.typing import NDArray
from typing import TypeVar

# my type aliases
Filename = str
Mask = NDArray[bool]
Dict_keys = type({}.keys())

Particle = TypeVar('Particle')