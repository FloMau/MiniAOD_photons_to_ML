import numpy as np
import argparse

from typing import List, Tuple
from numpy.typing import NDArray


def setup_parser() -> Tuple[str, str, str]:
    parser = argparse.ArgumentParser(description='convert list of failed jobs to filenames\n'
                                                 'this script will take a file with the failed jobnumbers, convert them'
                                                 ' to indices and return the corresponding filenames as a new file')
    parser.add_argument('failedlist', help='file with failed jobs')
    parser.add_argument('datalist', help='file with all datafiles')
    parser.add_argument('outfile', help='name of outpufile')
    args = parser.parse_args()
    return args.failedlist, args.datalist, args.outfile

def find_number_in_str(text: str) -> int:
    '''finds number in str and converts it to an int, returns None if no number found'''
    maxtries: int = len(text)
    idx: int = 0
    while isinstance(text, str):
        if idx == maxtries: raise ValueError('No number found in str')
        try:
            number: int = int(text[idx:])
        except ValueError:
            idx += 1
    return number


## see description in setup_parser
failedlist, datalist, outfile = setup_parser()
failed: NDArray[str] = np.loadtxt(failedlist, dtype=str)
data: NDArray[str] = np.loadtxt(datalist, dtype=str)
numberlist: List = []

for name in failed:
    number: str = name.split('_')[-1]
    try:
        number: int = int(number)
    except ValueError:
        number: int = find_number_in_str(number)
    # by this time number got converted to int
    if number >= len(data):
        raise Exception('the datafile does not contain enough datafiles')
    numberlist += [number]

numberlist.sort()
out: NDArray[str] = data[numberlist]
with open(outfile, 'w') as file:
    file.write('\n'.join(out))

print('saved as:',  outfile)
print('FINISHED')





