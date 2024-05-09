import ROOT
ROOT.gROOT.SetBatch(True)

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Handle, Events

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT])
import argparse

from typing import List, Tuple, Optional, Union
from typing import TypeVar

# my type aliases
particle = TypeVar('particle')



def get_pt(photon: particle) -> float:
    return photon.pt()

def get_et(photon: particle) -> float:
    return photon.et()

def get_eta(photon: particle) -> float:
    return photon.eta()

def get_phi(photon: particle) -> float:
    return photon.phi()

def get_r9(photon: particle) -> float:
    return photon.full5x5_r9()

def get_HoE(photon: particle) -> float:
    return photon.hadronicOverEm()

def get_sigma_ieie(photon: particle) -> float:
    return photon.sigmaEtaEta()

def get_isolations(photon: particle) -> Tuple[float, float, float, float]:
    '''I_ch, I_gamma, I_n'''
    return photon.chargedHadronIso(), photon.photonIso(), photon.neutralHadronIso(), photon.trackIso()

def get_ecalIso(photon: particle) -> float:
    return photon.ecalPFClusterIso()

def get_hcalIso(photon: particle) -> float:
    return photon.hcalPFClusterIso()

def is_real(photon: particle) -> bool:
    '''returns True for a real photon and False for a fake'''
    try:
        pdgId = photon.genParticle().pdgId()
        if pdgId == 22:
            return True  # real
        else:
            return False  # fake
    except ReferenceError:
        return False  # fake

def did_convert_full(photon: particle) -> bool:
    '''checks if photon converted and both tracks got reconstructed'''
    if photon.conversions(): return True
    else: return False

def did_convert_oneleg(photon: particle) -> bool:
    '''checks if photon converted and only one track got reconstructed'''
    if photon.conversionsOneLeg(): return True
    else: return False

def get_detector_ID(photon: particle) -> bool:
    '''returns True for Barrel and False for Endcap'''
    return photon.superCluster().seed().seed().subdetId()==1

def pass_eveto(photon: particle) -> bool:
    return photon.passElectronVeto()

def get_mc_truth(photon: particle) -> int:
    try:
        pdgId = photon.genParticle().pdgId()
        return pdgId
    except ReferenceError:
        return -1

def get_bdt_run2(photon: particle) -> float:
    # mva is range -1 to 1, I use 0 to 1
    mva = photon.userFloat("PhotonMVAEstimatorRunIIFall17v2Values")
    return (mva+1)/2

def get_bdt_run3(photon: particle) -> float:
    # mva is range -1 to 1, I use 0 to 1
    mva = photon.userFloat("PhotonMVAEstimatorRunIIIWinter22v1Values")
    return (mva+1)/2

def get_all(photon: particle) -> dict[str, Union[int, float, bool]]:
    out = {
        'pt': get_pt(photon),
        'et': get_et(photon),
        'eta': get_eta(photon),
        'phi': get_phi(photon),
        'r9': get_r9(photon),
        'HoE': get_HoE(photon),
        'sigma_ieie': get_sigma_ieie(photon),
        'I_ch': get_isolations(photon)[0],
        'I_gamma': get_isolations(photon)[1],
        'I_n': get_isolations(photon)[2],
        'I_tr': get_isolations(photon)[3],
        'ecalIso': get_ecalIso(photon),
        'hcalIso': get_hcalIso(photon),
        'real': is_real(photon),
        'mc_truth': get_mc_truth(photon),
        'bdt2': get_bdt_run2(photon),
        'bdt3': get_bdt_run3(photon),
        'detID': get_detector_ID(photon),
        'converted': did_convert_full(photon),
        'convertedOneLeg': did_convert_oneleg(photon),
        'eveto': pass_eveto(photon)
    }
    return out


def main(file: str) -> pd.DataFrame:
    '''creates a dataframe looping through all events and photons per event in a given file'''
    print("INFO: opening file", file.split("/")[-1])
    print(file)
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    events = Events(file)

    data_list = []  # save data in nested list to convert to DataFrame later
    num_real_list = []
    num_fake_list = []
    for i, event in enumerate(events):
        print("\t INFO: processing event", i)
        event.getByLabel(photonLabel, photonHandle)

        num_real = 0
        num_fake = 0
        for photon in photonHandle.product():
            if is_real(photon): num_real += 1
            else: num_fake += 1
            data_list += [get_all(photon)]  # list of dicts with the values of the respective photon
        num_real_list += [num_real]*(num_real + num_fake)
        num_fake_list += [num_fake]*(num_real + num_fake)
    print('INFO: all events processed')

    df = pd.DataFrame(data_list)  # labels are taken from the dicts in data_list
    df['num_real'] = num_real_list
    df['num_fake'] = num_fake_list
    return df


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='python script to create a dataframe with all relevant photon values from given file\n'
                                                 'output is a .pkl in /net/scratch_cms3a/kappe/data',
                                     prog='get_photon_values.py')
    parser.add_argument('filename', help='file to be read')
    parser.add_argument('--local', action='store_true', help='set to not access the grid '
                                                             '(i.e. you want to use a file on your own system)')
    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = setup_parser()
    args = parser.parse_args()
    file: str = args.filename
    local: bool = args.local

    if not local:
        file: str = ('root://xrootd-cms.infn.it/' + '/store/test/xrootd/' +
                     # 'T2_DE_DESY' failed
                     # 'T2_BE_IIHE' failed
                     # 'T2_CH_CSCS' failed
                     # 'T2_IT_Legnaro' failed
                     # 'T2_IT_Pisa' faield
                     # 'T2_IT_Rome' failed
                     'T2_US_Wisconsin'

                     + file)
    df: pd.DataFrame = main(file)

    savedir: str = '/net/scratch_cms3a/kappe/new_data/'
    outname: str = file.split('/')[-1].split('.')[0]  # name of file without directory und ending
    savename: str = savedir + outname + '.pkl'
    df.to_pickle(savename)
    print('INFO: file saved as:', savename)
    print('FINISHED')
