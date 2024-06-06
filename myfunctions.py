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
    """ 
    creates an array of slices indexing one photon in the sparse arrays each
    with this you can select the sparse rechits from e.g. a single photon
    """
    idx_photon = sparse[1]
    idxs_sparse = np.unique(idx_photon, return_index=True, return_counts=False)[1]  
    slices = np.array([slice(idxs_sparse[i], idxs_sparse[i+1]) for i in range(len(idxs_sparse)-1)] + [slice(idxs_sparse[-1], len(idxs_sparse))])
    return slices

def slice_sparse(sparse: Sparse, mask: Mask, slice_array: Optional[NDArray] = None) -> Sparse:
    """
    This function slices sparse rechits based on specified conditions (e.g., pt > 50 or selecting fakes). 
    This function translates the condition and applies it to the sparse rechits.
    (Sparse rechits are about 25 times longer than the main dataframe on which most conditions apply)

    Parameters:
        sparse (ndarray): Sparse rechits to be sliced.
        mask (ndarray): Condition to be applied.
        slice_array (ndarray, optional): The result from `create_slice_arr`. 
                                        Use this if slicing more than once to avoid recomputation 
                                        (e.g., using different pt cuts).

    Example usage:
        df = pd.read_pickle(dataframefile)
        sparse = load_sparse(rechitsfile)

        # Select fakes
        mask = ~df.real.to_numpy()
        fake_rechits = slice_sparse(sparse, mask)

        # Apply different pt cuts
        slice_arr = get_slice_arr(sparse)
        cut50 = df.pt > 50
        cut100 = df.pt > 100

        rechits_50 = slice_sparse(sparse, cut50, slice_arr)
        rechits_100 = slice_sparse(sparse, cut100, slice_arr)  # Passing slice_arr saves about 30s after the first time
    """
    if slice_array is None:
        slice_array = create_slice_arr(sparse)
    selected_slices = slice_array[mask]
    sel = np.r_[tuple(selected_slices)]  # this converts the slices in an array of indices, which the slices would access
    selected: Sparse = tuple(arr[sel] for arr in sparse)
    return selected

