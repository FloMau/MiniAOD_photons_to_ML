import numpy as np
import pandas as pd

from numpy.typing import NDArray
from mytypes import Filename, Sparse

# rechit functions
def dense_to_sparse(dense: NDArray) -> Sparse:
    """convert dense rechit array to sparse representation
    the returned arrays are the values and the indices of the photon, row and column of the values in that order"""
    idxs_ = np.nonzero(dense)  # tuple of three arrays
    values_ = dense[idxs_]
    idxs_ = [idx_array.astype(np.int32) for idx_array in idxs_]
    return values_, *idxs_

def save_sparse_rechits(savename: Filename, sparse: Sparse) -> None:
    np.savez(savename, values=sparse[0], idx1=sparse[1], idx2 = sparse[2], idx3=sparse[3])
    print(f'INFO: file saved as {savename}')

