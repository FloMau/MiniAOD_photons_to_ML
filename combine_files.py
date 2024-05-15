import numpy as np
import argparse

import numpy as np
import pandas as pd
import argparse

from typing import List, Tuple, Optional, Sequence, Any
from numpy.typing import NDArray
from mytypes import Filename, Mask

def file_ending(filename: Filename) -> str:
    return filename.split('.')[-1]


def check_extensions(file_list: Sequence[Filename], extension: Optional[str] = None) -> None:
    """checks if all files in file_list have the same extension and returns said extension"""
    endings: List[str] = [file_ending(file) for file in file_list]
    if not endings.count(endings[0]) == len(endings):
        raise ValueError(f'not all file endings are the same')
    if extension is not None and endings[0] != extension:
        raise ValueError(f'endings in file list {endings[0]} do not match required file type {extension}')


def combine_dataframes(file_list: Sequence[Filename], savename: Filename) -> None:
    print(f'combining {len(file_list)} files')
    all_data: List[Any] = []  # to be appended
    labels: Optional[List[str]] = None
    for i, file in enumerate(file_list):
        print(f'{i+1:>3}/{len(file_list)}: {file}')
        all_data += pd.read_pickle(file).values.tolist()

        if labels is None:  # first file: read labels
            labels = pd.read_pickle(file).columns.tolist()

    df: pd.DataFrame = pd.DataFrame(all_data, columns=labels)
    print(f"INFO: saving as: {savename}")
    df.to_pickle(savename)


def combine_rechits(file_list: Sequence[Filename], savename: Filename) -> None:
    print(f'combining {len(file_list)} files')
    all_data: List = []  # to be appended
    for i, file in enumerate(file_list):
        print(f'{i+1:>3}/{len(file_list)}: {file}')
        data = np.memmap(file, dtype=np.float32, mode='r')  # cannot specify shape because I do not know the number of photons in the file yet
        # first 32 positions are meta information because I did not specify the shape
        # need to get rid of them and get the shape back to 32x32 images
        num_photons = int((len(data)-32)/32/32)
        shape = (num_photons, 32, 32)
        data = data[32:].reshape(shape)
        all_data += [data]
    
    large_array = np.vstack(all_data)
    print('files combined, starting saving...')
    np.save(savename, large_array)
    print(f"INFO: saved as: {savename}")


def process_args() -> Tuple[Sequence[Filename], Filename, str]:
    """set up the ArgParser, check if inputs are valid and do some processing"""
    parser = argparse.ArgumentParser(description='combine dataframes or rechits into one large file',
                                     prog='combine_files.py')
    parser.add_argument('files', nargs='+', help='file to be read')
    parser.add_argument('--outname', required=True, help='name of the outputfile')
    parser.add_argument('--list', action='store_true', help='use if input is a file with the filenames to be read from')

    args = parser.parse_args()
    files_: Sequence[Filename] = args.files
    outname_: Filename = args.outname
    if args.list:
        files_ = np.loadtxt(files_[0], dtype=str)  # files is list with one element
    ending_ = file_ending(outname_)
    check_extensions(files_, extension=ending_)
    return files_, outname_, ending_

if __name__ == "__main__":
    files, outname, ending = process_args()
    savedir: str = ''

    if ending == 'pkl':
        combine_dataframes(files, outname)
    elif ending == 'npy':
        combine_rechits(files, outname)
    else:
        raise ValueError(f'file ending not supported: {ending}')
    print('Finished')

