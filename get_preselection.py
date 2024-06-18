import numpy as np
import pandas as pd
import argparse

from typing import Tuple, Union
from mytypes import Filename, Mask

def process_parser() -> Tuple[Filename, Filename]:
    parser = argparse.ArgumentParser(description='pythonscript to calculate and save preselection mask')
    parser.add_argument('datafile', help='file to be read')
    parser.add_argument('outfile', help='save preselection as file')
    args = parser.parse_args()
    filename_: Filename = args.datafile
    outfile_: Filename = args.outfile
    return filename_, outfile_

def get_preselection(df: Union[pd.DataFrame, dict]) -> Mask:
    '''returns mask for preselction, without eveto'''
    pt = df['pt'] > 25  # no leading photon, because I do single photon studies
    transition = (1.44 < np.abs(df['eta'])) & (np.abs(df['eta']) < 1.57)
    eta = (np.abs(df['eta']) < 2.5) & (~transition)
    HoE = df['HoE'] < 0.08
    iso_gamma = df['I_gamma'] < 4.0
    iso_track = df['I_tr'] < 6.0

    # barrel
    barrel = df['detID'] == 1
    R9_small_barrel = df['r9'] > 0.5
    R9_large_barrel = df['r9'] > 0.85
    sigma_barrel = df['sigma_ieie'] < 0.015
    barrel1 = barrel & R9_small_barrel & sigma_barrel & iso_gamma & iso_track
    barrel2 = barrel & R9_large_barrel
    barrel = barrel1 | barrel2

    # endcap
    endcap = df['detID'] == 0
    R9_small_endcap = df['r9'] > 0.80
    R9_large_endcap = df['r9'] > 0.90
    sigma_endcap = df['sigma_ieie'] < 0.035
    endcap1 = endcap & R9_small_endcap & sigma_endcap & iso_gamma & iso_track
    endcap2 = endcap & R9_large_endcap
    endcap = endcap1 | endcap2

    # combine Masks
    shower_shape = HoE & (barrel | endcap)
    one_of = (df['r9'] > 0.8) | ((df['I_ch']/df['pt']) < 0.3) | (df['I_ch'] < 20)
    total_mask = pt & eta & shower_shape & one_of
    return total_mask

def get_total_preselection(df: Union[pd.DataFrame, dict]) -> Mask:
    """adds eveto to preselection and filters NaNs in rho"""
    return get_preselection(df) & df['eveto'] & (~np.isnan(df['rho']))


if __name__ == "__main__":
    filename, outfile = process_parser()

    df = pd.read_pickle(filename)
    preselection = get_total_preselection(df) & df['eveto']

    np.save(outfile, preselection)
    print(f'INFO: saved as {outfile}')

    print('FINISHED')

