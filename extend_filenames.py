"""appends the Aachen storage site prefix to the filennames in the given file"""
import numpy as np
import argparse


def setup_parser():
    parser = argparse.ArgumentParser(description="appends the Aachen storage site prefix to the filennames in the given file")
    parser.add_argument('filename', help='file to be read')
    parser.add_argument('--sitename', default='T2_DE_RWTH', help='site prefeix to be added, default is T2_DE_RWTH')
    return parser

def check_if_prefixed(name) -> bool:
    if not isinstance(name, str): name = name[0]  # if list of names is given, take the first one
    return name[:len('/store/test/xrootd/')] == '/store/test/xrootd/'

def remove_prefix(name: str):
    return ''.join(name.split('/')[5:])

def add_prefix(name: str, prefix: str):
    return prefix + name


remove_prefix = np.vectorize(remove_prefix)
add_prefix = np.vectorize(add_prefix)

def write_to_file(filename, output):
    with open(filename, 'w') as file:
        file.write('\n'.join(output))


def main(filename, sitename):
    names = np.loadtxt(filename, dtype=str)
    prefix = '/store/test/xrootd/' + sitename  # no / at the end because names start with it
    if check_if_prefixed(names):
        names = remove_prefix(names)
    names = add_prefix(names, prefix)
    write_to_file(filename, names)
    print('file saved as:', filename)
    print('FINISHED')


if __name__=='__main__':
    parser = setup_parser()
    args = parser.parse_args()
    main(args.filename, args.sitename)

