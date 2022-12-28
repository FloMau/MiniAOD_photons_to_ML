from CRABClient.UserUtilities import config

config = config()

config.General.requestName = 'Slimming'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'slim_MiniAODs.py'

config.Data.inputDataset = '/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/Run3Winter22MiniAOD-FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 10
config.Data.publication = False 
config.Data.allowNonValidInputDataset  = True
config.Data.outputDatasetTag = 'GJetRun3_Pt-40toInf'


config.Data.outLFNDirBase = "/store/group/phys_egamma/ec/fmausolf/GJet_Run3_MiniAOD/Pt-40toInf/"  #points to /eos/cms/store/group/phys_egamma/...
config.Data.publication = False
config.Site.storageSite = 'T2_CH_CERN'
config.Site.ignoreGlobalBlacklist = True
