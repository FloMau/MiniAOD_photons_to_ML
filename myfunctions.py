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

def load_sparse(file: Filename) -> Sparse:
    npz = np.load(file)
    sparse = (npz['values'], npz['idx1'], npz['idx2'], npz['idx3'])
    return sparse

def sparse_to_dense(sparse: Sparse, shape: Union[Tuple[int, int], Tuple[int, int, int]]=(32, 32)) -> NDArray:
    """shape can be 2d or 3d, 
    if 2d it must be the shape of the images and the number of photons will be inferred"""
    values, indices = sparse[0], sparse[1:]
    if len(shape)==2:
        num = np.max(indices[0])+1
        shape = (num, *shape)
    
    dense = np.zeros(shape, dtype=np.float32)
    dense[indices] = values
    return dense


def create_slice_arr(sparse: Sparse) -> NDArray:
    idx_photon = sparse[1]
    idxs_dense, _, counts = np.unique(idx_photon, return_index=True, return_counts=True)  
    # shape: 8Mio, range: 8Mio and shape 8Mio, range 0-32 with average 25
    idxs_sparse = np.cumsum(counts)  # shape 8Mio, range(200Mio)
    
    slices = np.array([slice(idxs_sparse[i], idxs_sparse[i]) for i in range(len(idxs_sparse))])
    return slices

def slice_sparse(sparse: Sparse, mask: Mask, slice_array: Optional[NDArray] = None) -> Sparse:
    if slice_array is None:
        slice_array = create_slice_arr(sparse)
    selected_slices = slice_array[mask]
    sel = np.r_[tuple(selected_slices)]  # this converts the slices in an array of indices, which the slices would access
    selected: Sparse = tuple(arr[sel] for arr in sparse)
    return selected





