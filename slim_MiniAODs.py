import FWCore.ParameterSet.Config as cms

process = cms.Process('NoSplit')

### to run on full dataset with CRAB3:
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring())
### to test locally: 
# process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("root://xrootd-cms.infn.it//store/mc/Run3Winter22MiniAOD/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/2430000/0176f78e-294f-4ec1-9e8d-ddcbf471ae65.root"))

# process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        "drop *", 
        "keep *_slimmedPhotons_*_*",
        "keep *_*_reducedSuperClusters_*", 
        "keep *PhotonCore*_*_*_*",
        "keep *CaloCluster*_*_*_*",
        "keep *_*_reducedEBRecHits_*",
        "keep *_prunedGenParticles_*_*",
        "keep *_slimmedGenJets_*_*"),
        # "keep *_packedPFCandidates_*_*"), # these take a lot of storage, but will be needed later for the tracks and isolation studies 
    fileName = cms.untracked.string('output.root'),
)
process.out = cms.EndPath(process.output)