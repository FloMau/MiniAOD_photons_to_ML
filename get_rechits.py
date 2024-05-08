import ROOT
ROOT.gROOT.SetBatch(True)

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Handle, Events

import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep
import copy
plt.style.use([hep.style.ROOT])
import os
import argparse

def select_recHits(recHits, photon_seed, distance=5):
    """
    This function selects ECAL RecHits around the seed of the photon candidate.
    Selects a square of size 2*distance+1
    """
    seed_i_eta = photon_seed.ieta()
    seed_i_phi = photon_seed.iphi()

    rechits_array = np.zeros((2 * distance + 1, 2 * distance + 1))
    for recHit in recHits:
        # get crystal indices to see if they are close to our photon
        raw_id = recHit.detid().rawId()
        ID = ROOT.EBDetId(raw_id)

        i_eta = ID.ieta()
        i_phi = ID.iphi()

        if abs(i_phi - seed_i_phi) > distance or abs(i_eta - seed_i_eta) > distance:
            continue

        rechits_array[i_eta - seed_i_eta + distance, i_phi - seed_i_phi + distance] = recHit.energy()
    return rechits_array


def main(file, distance=5):
    '''creates a dataframe looping through all events and photons per event in a given file'''
    print("INFO: opening file", file.split("/")[-1])
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    RecHitHandleEB, RecHitLabelEB = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits"
    RecHitHandleEE, RecHitLabelEE = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEERecHits"
    events = Events(file)

    data_list = []  # save data in nested list
    count = 0
    for i, event in enumerate(events):
        print("INFO: processing event", i)
        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabelEB, RecHitHandleEB)
        event.getByLabel(RecHitLabelEE, RecHitHandleEE)

        for photon in photonHandle.product():
            count += 1
            seed_id = photon.superCluster().seed().seed()
            seed_id = ROOT.EBDetId(seed_id)  # get crystal indices of photon candidate seed:
            # usung EEDetId gives the same value but errors in select_recHits becaause it has no attribute ieta

            if photon.superCluster().seed().seed().subdetId()==1:
                recHits = RecHitHandleEB.product()
            else:
                recHits = RecHitHandleEE.product()
            rechits_array = select_recHits(photon_seed=seed_id, recHits=recHits, distance=distance)

            data_list += [rechits_array]

    print('INFO: all events processed')
    data = np.array(data_list)
    return data


def setup_parser():
    parser = argparse.ArgumentParser(description='create dataframe with all relevant photon values from file', prog='make_dataframe.py')
    parser.add_argument('filename', help='file to be read')
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    file = args.filename

    savedir = '/net/scratch_cms3a/kappe/new_rechits/'
    outname = file.split('/')[-1].split('.')[0] # name of file without directory und ending

    data = main('root://xrootd-cms.infn.it/' + '/store/test/xrootd/' + 'T2_US_Wisconsin' + file)

    savename = savedir + outname + '.npy'
    np.save(savename, data)
    print('file saved as:', savename)
    print('FINISHED')


