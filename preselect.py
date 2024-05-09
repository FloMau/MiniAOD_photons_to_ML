import numpy as np
import pandas as pd
import argparse
from abc import abstractmethod

from typing import Tuple, Optional
from numpy.typing import NDArray
from mytypes import Filename


class FileHandler:
    @abstractmethod
    def load(self, filename: Filename):
        pass

    @abstractmethod
    def save(self, data_, savename: Filename):
        pass


class NumpyFileHandler(FileHandler):
    def load(self, filename) -> NDArray:
        return np.load(filename)

    def save(self, data_: NDArray, savename) -> None:
        np.save(savename, data_)
        print(f'INFO: file saved as {savename}')


class PandasFileHandler(FileHandler):
    def load(self, filename: Filename) -> pd.DataFrame:
        return pd.read_pickle(filename)

    def save(self, data_: pd.DataFrame, savename: Filename) -> None:
        pd.to_pickle(data_, savename)
        print(f'INFO: file saved as {savename}')


def get_appropiate_handler(filename: Filename) -> FileHandler:
    ending = filename.split('.')[-1].strip()
    if ending == 'npy':
        return NumpyFileHandler()
    elif ending == 'pkl':
        return PandasFileHandler()
    else:
        raise ValueError(f'filetype "{ending}" not recognised or supported')

def process_args() -> Tuple[FileHandler, Filename, Filename, Filename]:
    parser = argparse.ArgumentParser(description='pythonscript to save a preselected version of dataset')
    parser.add_argument('datafile', help='file to be preselected')
    parser.add_argument('preselectionfile', help='file with preselection')
    parser.add_argument('--outfile', default=None, help='new filename, default appends "_preselected" to inputfilename')

    args = parser.parse_args()
    datafile_: Filename = args.datafile
    preselectionfile_: Filename = args.preselectionfile
    outfile_: Optional[Filename] = args.outfile

    handler = get_appropiate_handler(datafile_)

    if outfile_ is None:
        name, ending = datafile_.split('.')
        outfile_ = name + '_preselected' + '.' + ending
    if outfile_ == datafile_:
        raise ValueError("outfile must not be  datafile")

    return handler, datafile_, preselectionfile_, outfile_


if __name__ == "__main__":
    filehandler, datafile, preselectionfile, outfile = process_args()
    # filehandler loads the file regardless of filetype

    data = filehandler.load(datafile)
    preselection = np.load(preselectionfile)
    new_data = data[preselection]

    filehandler.save(new_data, outfile)
    print("FINISHED")
