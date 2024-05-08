import ROOT
from DataFormats.FWLite import Handle, Events

import numpy as np
import pandas as pd
import argparse

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from typing import TypeVar

# my type aliases
Filename = str
Mask = NDArray[np.bool_]
Particle = TypeVar('Particle')


ROOT.gROOT.SetBatch(True)
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()


def get_detector_ID(photon: Particle) -> bool:
    '''returns True for Barrel and False for Endcap'''
    return photon.superCluster().seed().seed().subdetId()==1

def main(file: Filename) -> List[float]:
    """loop through all events and photons per event in a given file"""
    print("INFO: opening file", file.split("/")[-1])
    print('full filename:', file)
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    pileupHandle, pileupLabel = Handle("double"), "fixedGridRhoAll"
    events = Events(file)

    pileup_list = []
    for i, event in enumerate(events):
        if i == 0: print("\tINFO: file open sucessful, starting Event processing")
        # print("\t INFO: processing event", i)
        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(pileupLabel, pileupHandle)

        for photon in photonHandle.product():
            # skip endcap
            if not get_detector_ID(photon): continue
            pileup = pileupHandle.product()[0]
            pileup_list += [pileup]


    print('INFO: all events processed')
    return pileup_list


def process_args() -> int:
    parser = argparse.ArgumentParser(description='python script to get the pileup from the files in jobs/datafiles.txt',
                                     prog='preprocess.py')
    parser.add_argument('segment', help='number between 1 and 5. only that segment of the filelist will be processed')
    args = parser.parse_args()

    print(args)
    segment_: int = int(args.segment)
    return segment_


if __name__ == "__main__":
    segment = process_args()
    file_list = np.genfromtxt('jobs/datafiles.txt', dtype=str)
    segment = slice(50*(segment-1), 50*segment)
    file_list = file_list[segment]
    pileup_list = []

    for i, file in enumerate(file_list):
        file = 'root://xrootd-cms.infn.it/' + '/store/test/xrootd/' + 'T2_US_Wisconsin' + file
        pileup_list += main(file)
        print(f'processed file number {i+1} \n')

    # save stuff
    outname: Filename = f'data/pileup_{segment}.npy'
    np.save(outname, pileup_list)
    print('INFO: pileup saved as:', outname)
    print('FINISHED')
