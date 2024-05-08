import ROOT
from DataFormats.FWLite import Handle, Events

import numpy as np
import pandas as pd
import argparse

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from typing import TypeVar
# from mytypes import Filename, Mask, Particle
import os

Filename = str
Mask = NDArray[np.bool_]

Particle = TypeVar('Particle')

ROOT.gROOT.SetBatch(True)
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()


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

    try:
        pdgId = photon.genParticle().pdgId()
        real = True if pdgId == 22 else False
        true_energy = photon.genParticle().energy()
    except ReferenceError:
        real = False  # fake
        true_energy = -999.

    try:
        mc_truth = photon.genParticle().pdgId()
    except ReferenceError:
        mc_truth = -1

    return {
        'pt': photon.pt(),
        'et': photon.et(),
        'eta': photon.eta(),
        'phi': photon.phi(),
        'r9': photon.full5x5_r9(),
        'HoE': photon.hadronicOverEm(),
        'sigma_ieie': photon.sigmaEtaEta(),
        'I_ch': photon.chargedHadronIso(),
        'I_gamma': photon.photonIso(),
        'I_n': photon.neutralHadronIso(),
        'I_tr': photon.trackIso(),
        'ecalIso': photon.ecalPFClusterIso(),
        'hcalIso': photon.hcalPFClusterIso(),
        'real': real,
        # 'mc_truth': mc_truth,  # Uncomment if mc_truth is needed despite the comment in the original code
        'bdt2': (photon.userFloat("PhotonMVAEstimatorRunIIFall17v2Values") + 1) / 2,
        'bdt3': (photon.userFloat("PhotonMVAEstimatorRunIIIWinter22v1Values") + 1) / 2,
        'detID': True if photon.superCluster().seed().seed().subdetId() == 1 else False,
        'converted': True if photon.conversions() else False,
        'convertedOneLeg': True if photon.conversionsOneLeg() else False,
        'eveto': photon.passElectronVeto(),
        "true_energy": true_energy,
        "SC_raw_energy": photon.superCluster().rawEnergy()
    }


def select_rechits(recHits, photon_seed, distance=5) -> NDArray[float]:
    """
    This function selects ECAL RecHits around the seed of the photon candidate.
    Selects a square of size 2*distance+1
    """
    seed_i_eta = photon_seed.ieta()
    seed_i_phi = photon_seed.iphi()

    rechits_array: NDArray[int] = np.zeros((2 * distance + 1, 2 * distance + 1))
    for recHit in recHits:
        # get crystal indices to see if they are close to our photon
        raw_id = recHit.detid().rawId()
        ID = ROOT.EBDetId(raw_id)

        i_eta: int = ID.ieta()
        i_phi: int = ID.iphi()

        if abs(i_phi - seed_i_phi) > distance or abs(i_eta - seed_i_eta) > distance:
            continue

        rechits_array[i_eta - seed_i_eta + distance, i_phi - seed_i_phi + distance] = recHit.energy()

    if distance % 2 == 0:
        # Calculate the sum of energies for the outer rows and columns
        row_sums = rechits_array.sum(axis=1)  # Sum of each row
        col_sums = rechits_array.sum(axis=0)  # Sum of each column

        # Determine which row and column to remove
        min_row = np.argmin([row_sums[0], row_sums[-1]])
        min_col = np.argmin([col_sums[0], col_sums[-1]])

        # Remove the row and column with the minimum energy sum
        if min_row == 0:
            rechits_array = rechits_array[1:]  # Remove first row
        else:
            rechits_array = rechits_array[:-1]  # Remove last row

        if min_col == 0:
            rechits_array = rechits_array[:, 1:]  # Remove first column
        else:
            rechits_array = rechits_array[:, :-1]  # Remove last column

    return rechits_array


def main(file: Filename, rechitdistance: int = 5) -> Tuple[pd.DataFrame, NDArray[NDArray[float]]]:
    """loop through all events and photons per event in a given file"""
    print("INFO: opening file", file.split("/")[-1])
    print('full filename:', file)
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    RecHitHandleEB, RecHitLabelEB = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits"
    RecHitHandleEE, RecHitLabelEE = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEERecHits"
    rhoHandle, rhoLabel = Handle("std::double"), "fixedGridRhoAll"
    events = Events(file)

    # lists to fill in the eventloop:
    df_list: List[dict] = []  # save data in nested list to convert to DataFrame later
    rechit_list: List[NDArray] = []  # save data in nested list to convert to DataFrame later
    num_real_list: List[int] = []
    num_fake_list: List[int] = []
    for i, event in enumerate(events):
        if i == 0:
            print("\tINFO: file open sucessful, starting Event processing")
        elif i % 10_000 == 0:
            print(f"\tINFO: processing event {i}.")
        # print("\t INFO: processing event", i)
        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabelEB, RecHitHandleEB)
        event.getByLabel(RecHitLabelEE, RecHitHandleEE)
        event.getByLabel(rhoLabel, rhoHandle)

        num_real: int = 0
        num_fake: int = 0
        for photon in photonHandle.product():
            if not get_detector_ID(photon): continue
            # dataframe
            if is_real(photon):
                num_real += 1
            else:
                num_fake += 1
            
            seed_id = photon.superCluster().seed().seed()
            seed_id = ROOT.EBDetId(seed_id)  # get crystal indices of photon candidate seed:

            photonAttributes = get_all(photon)
            photonAttributes["rho"] = rhoHandle.product()[0]
            photonAttributes["seed_ieta"] = seed_id.ieta()
            photonAttributes["seed_iphi"] = seed_id.iphi()
            df_list += [photonAttributes]  # list of dicts with the values of the respective photon

            # rechits
            # usung photon.EEDetId() directly gives the same value but errors in select_recHits
            # because it has no attribute ieta
            if photon.superCluster().seed().seed().subdetId() == 1:
                recHits = RecHitHandleEB.product()
            else:
                recHits = RecHitHandleEE.product()
            rechits_array = select_rechits(photon_seed=seed_id, recHits=recHits, distance=rechitdistance)
            rechit_list += [rechits_array]

        num_real_list += [num_real]*(num_real + num_fake)
        num_fake_list += [num_fake]*(num_real + num_fake)
    print('INFO: all events processed')

    df: pd.DataFrame = pd.DataFrame(df_list)  # labels are taken from the dicts in data_list
    df['num_real'] = num_real_list
    df['num_fake'] = num_fake_list

    rechits = np.array(rechit_list, dtype=np.float32)
    return df, rechits


def process_file(file: Filename) -> None:

    datasite = 'T2_US_Wisconsin'
    datasite = 'T1_US_FNAL_Disk'
    if datasite is not None:
        file = '/store/test/xrootd/' + datasite + file
    file = 'root://xrootd-cms.infn.it/' + file

    df, rechits = main(file, rechitdistance=16)

    # save stuff
    savedir = '/net/scratch_cms3a/kappe/output07May2024_low_pt/'
    if not (os.path.exists(savedir + "/recHits") and  os.path.exists(savedir + "/df")):
        os.makedirs(savedir + "/df")
        os.makedirs(savedir + "/recHits")

    outname: str = file.split('/')[-1].split('.')[0]  # name of input file without directory and ending

    dfname: Filename = savedir + 'df/' + outname + '.pkl'
    df.to_pickle(dfname)
    print('INFO: photon df file saved as:', dfname)

    rechitname: str = savedir + 'recHits/' + outname + '.npy'
    np.save(rechitname, rechits)
    print('INFO: recHits file saved as:', rechitname)

    print('INFO: finished running.')

if __name__ == '__main__':
    process_file('/store/mc/Run3Summer22EEMiniAODv4/GJet_PT-40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/30000/cb93eb36-cefb-4aea-97aa-fcf8cd72245f.root')