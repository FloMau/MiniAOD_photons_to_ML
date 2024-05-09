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


from get_preselection import get_preselection

def get_pt(photon: Particle) -> float:
    return photon.pt()

def get_et(photon: Particle) -> float:
    return photon.et()

def get_eta(photon: Particle) -> float:
    return photon.eta()

def get_phi(photon: Particle) -> float:
    return photon.phi()

def get_r9(photon: Particle) -> float:
    return photon.full5x5_r9()

def get_HoE(photon: Particle) -> float:
    return photon.hadronicOverEm()

def get_sigma_ieie(photon: Particle) -> float:
    return photon.sigmaEtaEta()

def get_isolations(photon: Particle) -> Tuple[float, float, float, float]:
    """I_ch, I_gamma, I_n, I_track"""
    return photon.chargedHadronIso(), photon.photonIso(), photon.neutralHadronIso(), photon.trackIso()

def get_ecalIso(photon: Particle) -> float:
    return photon.ecalPFClusterIso()

def get_hcalIso(photon: Particle) -> float:
    return photon.hcalPFClusterIso()

def is_real(photon: Particle) -> bool:
    """returns True for a real photon and False for a fake"""
    try:
        pdgId = photon.genParticle().pdgId()
        if pdgId == 22:
            return True  # real
        else:
            return False  # fake
    except ReferenceError:
        return False  # fake

def did_convert_full(photon: Particle) -> bool:
    """checks if photon converted and both tracks got reconstructed"""
    if photon.conversions(): return True
    else: return False

def did_convert_oneleg(photon: Particle) -> bool:
    """checks if photon converted and only one track got reconstructed"""
    if photon.conversionsOneLeg(): return True
    else: return False

def get_detector_ID(photon: Particle) -> bool:
    '''returns True for Barrel and False for Endcap'''
    return photon.superCluster().seed().seed().subdetId()==1

def pass_eveto(photon: Particle) -> bool:
    return photon.passElectronVeto()

def get_mc_truth(photon: Particle) -> int:
    try:
        pdgId = photon.genParticle().pdgId()
        return pdgId
    except ReferenceError:
        return -1

def get_bdt_run2(photon: Particle) -> float:
    # mva is range -1 to 1, I use 0 to 1
    mva = photon.userFloat("PhotonMVAEstimatorRunIIFall17v2Values")
    return (mva+1)/2

def get_bdt_run3(photon: Particle) -> float:
    # mva is range -1 to 1, I use 0 to 1
    mva = photon.userFloat("PhotonMVAEstimatorRunIIIWinter22v1Values")
    return (mva+1)/2

def get_all(photon: Particle) -> dict[str, Union[int, float, bool]]:
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
        # 'mc_truth': get_mc_truth(photon),  # useless, because only photons are in the slimmed photon collection
        'bdt2': get_bdt_run2(photon),
        'bdt3': get_bdt_run3(photon),
        'detID': get_detector_ID(photon),
        'converted': did_convert_full(photon),
        'convertedOneLeg': did_convert_oneleg(photon),
        'eveto': pass_eveto(photon)
    }
    return out

def select_rechits(recHits, photon_seed, distance_phi: int, distance_eta: int) -> NDArray[float]:
    """
    This function selects ECAL RecHits around the seed of the photon candidate.
    Selects a rectangle of size (2*distance_phi+1)*(2*distance_eta+1)
    """
    seed_i_eta = photon_seed.ieta()
    seed_i_phi = photon_seed.iphi()

    rechits_array: NDArray[int] = np.zeros((2 * distance_eta + 1, 2 * distance_phi + 1))
    for recHit in recHits:
        # get crystal indices to see if they are close to our photon
        raw_id = recHit.detid().rawId()
        ID = ROOT.EBDetId(raw_id)

        i_eta: int = ID.ieta()
        i_phi: int = ID.iphi()

        if abs(i_phi - seed_i_phi) > distance_phi or abs(i_eta - seed_i_eta) > distance_eta:
            continue

        rechits_array[i_eta - seed_i_eta + distance_eta, i_phi - seed_i_phi + distance_phi] = recHit.energy()
    return rechits_array


def main(file: Filename, rechitdistance_phi: int, rechitdistance_eta: int, barrel_only: bool = False
         ) -> Tuple[pd.DataFrame, NDArray[NDArray[float]]]:
    """loop through all events and photons per event in a given file"""
    print("INFO: opening file", file.split("/")[-1])
    print('full filename:', file)
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    RecHitHandleEB, RecHitLabelEB = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits"
    RecHitHandleEE, RecHitLabelEE = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEERecHits"
    pileupHandle, pileupLabel = Handle("double"), "fixedGridRhoAll"
    events = Events(file)

    # lists to fill in the eventloop:
    df_list: List[dict] = []  # save data in nested list to convert to DataFrame later
    rechit_list: List[NDArray] = []  # save data in nested list to convert to DataFrame later
    num_real_list: List[int] = []
    num_fake_list: List[int] = []
    pileup_list: List[float] = []
    for i, event in enumerate(events):
        if i == 0: print("\tINFO: file open sucessful, starting Event processing")
        # print("\t INFO: processing event", i)
        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabelEB, RecHitHandleEB)
        event.getByLabel(RecHitLabelEE, RecHitHandleEE)
        event.getByLabel(pileupLabel, pileupHandle)

        num_real: int = 0
        num_fake: int = 0
        for photon in photonHandle.product():
            if barrel_only:
                if not get_detector_ID(photon): continue
            
            # stuff I have to count here
            if is_real(photon): num_real += 1
            else: num_fake += 1
            rho = pileupHandle.product()[0]
            # pileup_list += [pileup]

            # dataframe
            # df_list += [get_all(photon)]  # list of dicts with the values of the respective photon
            photon_values: dict = get_all(photon)
            photon_values["rho"] = rho
            if not get_preselection(photon_values): continue
            df_list += [photon_values]

            # rechits
            seed_id = photon.superCluster().seed().seed()
            seed_id = ROOT.EBDetId(seed_id)  # get crystal indices of photon candidate seed:
            # usung photon.EEDetId() directly gives the same value but errors in select_recHits
            # because it has no attribute ieta
            if photon.superCluster().seed().seed().subdetId() == 1:
                recHits = RecHitHandleEB.product()
            else:
                recHits = RecHitHandleEE.product()
            rechits_array = select_rechits(photon_seed=seed_id, recHits=recHits, 
                                           distance_phi=rechitdistance_phi, distance_eta=rechitdistance_eta)
            rechit_list += [rechits_array]

        num_real_list += [num_real]*(num_real + num_fake)
        num_fake_list += [num_fake]*(num_real + num_fake)
    print('INFO: all events processed')

    df: pd.DataFrame = pd.DataFrame(df_list)  # labels are taken from the dicts in data_list
    df['num_real'] = num_real_list
    df['num_fake'] = num_fake_list
    # df['rho'] = pileup_list

    rechits = np.array(rechit_list, dtype=np.float32)
    return df, rechits


def process_args() -> Tuple[Filename, bool, bool]:
    parser = argparse.ArgumentParser(description='python script to create a dataframe with all relevant photon values from given file\n'
                                                 'output is a .pkl in /net/scratch_cms3a/kappe/data',
                                     prog='preprocess.py')
    parser.add_argument('filename', help='file to be read')
    location = parser.add_mutually_exclusive_group(required=False)
    location.add_argument('--local', action='store_true', help='set to not access the grid '
                                                               '(i.e. you want to use a file on your own system)')
    location.add_argument('--datasite', help='use to access a specific datasite')
    parser.add_argument('--test', action='store_true', help='set to save output file as "test.{pkl,root}"')
    parser.add_argument('--barrel_only', action='store_true', help='set to only save photons the barrel')
    # parser.add_argument('--preselect', action='store_true', help='set to only save photons which pass preselection')
    args = parser.parse_args()
    print(args)
    file_: str = args.filename
    local: bool = args.local
    datasite: Optional[str] = args.datasite
    test_: bool = args.test
    barrel_only_: bool = args.barrel_only

    if datasite is not None:
        file_ = '/store/test/xrootd/' + datasite + file_
    if not local:
        file_ = 'root://xrootd-cms.infn.it/' + file_
    # if test_:

    return file_, test_, barrel_only_


if __name__ == "__main__":
    file, test, barrel_only = process_args()
    df, rechits = main(file, rechitdistance_phi=32, rechitdistance_eta=20, barrel_only=barrel_only)

    # save stuff
    savedir: str = '/net/scratch_cms3a/kappe/'
    outname: str = file.split('/')[-1].split('.')[0]  # name of file without directory und ending
    dfname: Filename = savedir + 'data/' + outname + '.pkl'
    rechitname: Filename = savedir + 'rechits/' + outname + '.npy'

    if test:
        dfname = 'data/test.pkl'
        rechitname = 'data/test.npy'
    df.to_pickle(dfname)
    print('INFO: file saved as:', dfname)
    np.save(rechitname, rechits)
    print('INFO: file saved as:', rechitname)
    print('FINISHED')
