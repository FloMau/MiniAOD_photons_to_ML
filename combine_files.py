import numpy as np
import pandas as pd
import argparse

from typing import List, Tuple, Optional, Sequence, Any
from numpy.typing import NDArray
from mytypes import Filename, Mask

class Preselector:
    def __init__(self, preselectionfile: Filename):
        self.preselection: Mask = np.load(preselectionfile)
        self.current_idx: int = 0
        self.max_idx: int = len(self.preselection) - 1

    def _update_idx(self, idx: int):
        self.current_idx = idx

    def get_part(self, start: int, stop: int) -> Mask:
        return self.preselection[start:stop]

    def get_next_segment(self, steps: int) -> Mask:
        next_index: int = self.current_idx + steps
        out = self.get_part(self.current_idx, next_index)
        self._update_idx(next_index)
        return out


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
    all_data: List[NDArray] = []  # to be appended
    for i, file in enumerate(file_list):
        print(f'{i+1:>3}/{len(file_list)}: {file}')
        all_data += np.load(file).tolist()
    print(f"INFO: saving as: {savename}")
    np.save(savename, np.array(all_data))


def combine_rechits_preselected(file_list: Sequence[Filename], savename: Filename, preselector: Preselector) -> None:
    print(f'combining {len(file_list)} files')
    all_data: List[NDArray] = []  # to be appended
    for i, file in enumerate(file_list):
        print(f'{i+1:>3}/{len(file_list)}: {file}')
        new: NDArray = np.load(file)
        current_mask = preselector.get_next_segment(len(new))
        all_data += new[current_mask].tolist()
    print(f"INFO: saving as: {savename}")
    np.save(savename, np.array(all_data))


def process_args() -> Tuple[Sequence[Filename], Filename, str, Optional[Preselector]]:
    """set up the ArgParser, check if inputs are valid and do some processing"""
    parser = argparse.ArgumentParser(description='combine dataframes or rechits into one large file',
                                     prog='combine_files.py')
    parser.add_argument('files', nargs='+', help='file to be read')
    parser.add_argument('--outname', required=True, help='name of the outputfile')
    parser.add_argument('--list', action='store_true', help='use if input is a file with the filenames to be read from')
    parser.add_argument('--preselection', help='filename, save only what matches the given preseection')

    args = parser.parse_args()
    files_: Sequence[Filename] = args.files
    outname_: Filename = args.outname
    presectionfile: Optional[Filename] = args.preselection
    if args.list:
        files_ = np.loadtxt(files_[0], dtype=str)  # files is list with one element
    ending_ = file_ending(outname_)
    check_extensions(files_, extension=ending_)

    preselection_: Optional[Preselector]
    if presectionfile is not None:
        preselection_ = Preselector(presectionfile)
    else:
        preselection_ = None
    return files_, outname_, ending_, preselection_


if __name__ == "__main__":
    files, outname, ending, preselection = process_args()
    # home_dir: str = "/home/home1/institut_3a/kappe/"
    # savedir: str = home_dir + "work/data/"
    # todo hardcoding is not smart

    if ending == 'pkl':
        combine_dataframes(files, outname)
    elif ending == 'npy':
        if preselection is None: combine_rechits(files, outname)
        else: combine_rechits_preselected(files, outname, preselection)
    else:
        raise ValueError(f'file ending not supported: {ending}')
    print('Finished')
