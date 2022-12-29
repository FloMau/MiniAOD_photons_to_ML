import ROOT
ROOT.gROOT.SetBatch(True)

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Handle, Events

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep
import copy 
plt.style.use([hep.style.ROOT])
import os 


def plot_image(image, title, path):

    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under('w')

    image[image<1e-6]=1e-6
    im = plt.imshow(image, norm=colors.LogNorm(vmin=0.01, vmax=image.max()), cmap=cmap, interpolation=None)
  
    plt.colorbar(im, label='Energy deposition [GeV]')
    plt.xlabel("iphi")
    plt.ylabel("ieta")
    plt.title(title)
    plt.savefig(path + ".pdf")
    plt.clf()

def hadronic_fake_candidate(photon_candidate, genJets, genParticles):
    """
    This function checks on truth level if the photon candidate stems from a jet. 
    Loops through all genJets and genParticles and checks if 
    the closest object at generator level is a prompt photon, prompt electron, prompt muon or stems from a jet. 
    Returns True / False 
    """

    min_DeltaR = float(99999)

    photon_vector = ROOT.TLorentzVector()
    photon_vector.SetPtEtaPhiE(photon_candidate.pt(), photon_candidate.eta(), photon_candidate.phi(), photon_candidate.energy())

    # print("\t\t Photon gen particle PDG ID:", photon_candidate.genParticle().pdgId())

    # this jet loop might be not needed... check later, but doesn't harm at this point 
    jet_around_photon = False 
    for genJet in genJets:
        # build four-vector to calculate DeltaR to photon 
        genJet_vector = ROOT.TLorentzVector()
        genJet_vector.SetPtEtaPhiE(genJet.pt(), genJet.eta(), genJet.phi(), genJet.energy())

        DeltaR = photon_vector.DeltaR(genJet_vector)
        # print("\t\t INFO: gen jet eta, phi, delta R ", genJet.eta(), genJet.phi(), DeltaR)
        if DeltaR < 0.3: 
            jet_around_photon = True 

    is_prompt = False
    pdgId = 0 

    for genParticle in genParticles:

        if genParticle.pt() < 1: continue # threshold of 1GeV for interesting particles 

        # build four-vector to calculate DeltaR to photon 
        genParticle_vector = ROOT.TLorentzVector()
        genParticle_vector.SetPtEtaPhiE(genParticle.pt(), genParticle.eta(), genParticle.phi(), genParticle.energy())
        
        DeltaR = photon_vector.DeltaR(genParticle_vector)
        # print("\t\t INFO: gen particle eta, phi, delta R ", genParticle.eta(), genParticle.phi(), DeltaR)
        if DeltaR < min_DeltaR and DeltaR < 0.3: 
            min_DeltaR = DeltaR
            pdgId = genParticle.pdgId()
            is_prompt = genParticle.isPromptFinalState()
        # print("\t\t INFO: PDG ID:", pdgId)

    prompt_electron = True if (abs(pdgId)==11 and is_prompt) else False 
    prompt_photon = True if (pdgId==22 and is_prompt) else False
    prompt_muon = True if (abs(pdgId)==13 and is_prompt) else False 
    
    if jet_around_photon and not (prompt_electron or prompt_photon or prompt_muon):
        return True
    else:
        return False


def select_recHits(recHits, photon_seed, distance=5):

    """
    This function selects ECAL RecHits around the seed of the photon candidate.
    Selects a square of size 2*distance+1
    """

    seed_i_eta = photon_seed.ieta()
    seed_i_phi = photon_seed.iphi()

    rechits_array = np.zeros((2*distance+1, 2*distance+1))
    for recHit in recHits:
        
        # get crystal indices to see if they are close to our photon 
        raw_id = recHit.detid().rawId()
        ID = ROOT.EBDetId(raw_id)
        i_eta = ID.ieta()
        i_phi = ID.iphi()

        if abs(i_phi-seed_i_phi) > distance or abs(i_eta-seed_i_eta) > distance:
            continue
        
        rechits_array[i_eta-seed_i_eta+distance, i_phi-seed_i_phi+distance] = recHit.energy()

    return rechits_array 





def main(path = "", distance=5):

    # object collections we want to read:
    # can look into files via: "edmDumpEventContent filepath" to show all available collections
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    RecHitHandle, RecHitLabel = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits" 
    genParticlesHandle, genParticlesLabel = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"
    genJetsHandle, genJetsLabel = Handle("std::vector<reco::GenJet>"), "slimmedGenJets"

    print("INFO: opening file", path.split("/")[-1])
    events = Events(path)

    # just read some events for testing 
    stop_index = 100
    for i,event in enumerate(events):

        if i == stop_index: 
            break 
    
        if i % 1000 == 0:
            print("\n \t INFO: processing event", i)

        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabel, RecHitHandle)
        event.getByLabel(genParticlesLabel, genParticlesHandle)
        event.getByLabel(genJetsLabel, genJetsHandle)

        genJets = genJetsHandle.product()
        genParticles = genParticlesHandle.product()
    
        for photon in photonHandle.product():

            is_real = False 
            is_hadronic_fake = False 

            if photon.pt() < 20: continue 
            #### to be implemented here: preselection criteria on R9, H/E, iso 
            #### if photon does not fulfill them: continue 

            seed_id = photon.superCluster().seed().seed()
            # only use photon candidates in the ECAL barrel (EB) at this point  
            if seed_id.subdetId() != 1: continue 
            
            # photon.genParticle seems to exist only if it is matched to a gen-level photon
            try: 
                pdgId = photon.genParticle().pdgId()
                if pdgId == 22: 
                    is_real = True 
           
            except ReferenceError:
                is_hadronic_fake = hadronic_fake_candidate(photon, genJets, genParticles)

            # get crystal indices of photon candidate seed:
            seed_id = ROOT.EBDetId(seed_id)

            recHits = RecHitHandle.product()

            rechits_array = select_recHits(photon_seed=seed_id, recHits=recHits, distance=distance)

            ### calo image can be plotted:
            # plot_image(rechits_array, str(i), str(i))

            ### to be implemented here: separately save real and fake photons for later ML studies



if __name__ == "__main__":

    # DoublePhotonGun samples 
    # path_test_file = "/eos/cms/store/group/phys_egamma/ec/fmausolf/DoublePhotonMiniAOD/01357d93-bd85-4317-87c4-e0f57a10c50a.root"

    # GJet samples (Run 3)
    # path_test_file = "/eos/cms/store/group/phys_egamma/ec/fmausolf/GJet_Run3_MiniAOD/0176f78e-294f-4ec1-9e8d-ddcbf471ae65.root"
    
    path_GJet_files = "/eos/cms/store/group/phys_egamma/ec/fmausolf/GJet_Run3_MiniAOD/Pt-40toInf/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/GJetRun3_Pt-40toInf/221228_143544/0000/"

    filenames = os.listdir(path_GJet_files)
    # just take one file for tests
    filenames = filenames[:1]

    for filename in filenames: 

        main(path = path_GJet_files+filename, distance=5)
