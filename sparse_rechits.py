import numpy as np
import pandas as pd
import argparse

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask


def process_args() -> str:
    parser = argparse.ArgumentParser(description='python script to create a dataframe with all relevant photon values from given file\n'
                                                 'output is a .pkl in /net/scratch_cms3a/kappe/data',
                                     prog='preprocess.py')
    parser.add_argument('filename', help='file to be read')
    args = parser.parse_args()
    file_: str = args.filename
    return file_

def convert_to_sparse(rechits: NDArray[float]) -> NDArray[Tuple[Tuple[int, int], float]]:
    """load full image rechits from file and convrt to (index, value) pair
    rechits is assumed to have dimension (N, image_len, image_len) i.e. a lot of images"""
    non_zero_idx: NDArray[int, int, int] = np.argwhere(rechits)  # incices, shape: (long, 3)
    non_zero: Mask = rechits != 0
    values = rechits[non_zero]

    image_shape = rechits.shape[-2:]
    new = []
    for (k, i, j), value in np.ndenumerate(rechits):
        if value==0: continue
        else: new += ((i, j), value)


    new = np.array()
    return



