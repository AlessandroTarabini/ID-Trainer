# In this file you can specify the training configuration
#####################################################################
######Do not touch this

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
#####################################################################
####Start here
#####################################################################
OutputDirName = 'SimpleBinaryClassification_XGBoost' #All plots, models, config file will be stored here
Debug=False # If True, only a small subset of events/objects are used for either Signal or background #Useful for quick debugging

#Branches to read #Should be in the root files #Only the read branches can be later used for any purpose
# branches=["scl_eta","ele*","matched*","EleMVACats",'passElectronSelection','Fall*']
branches=['pho*', 'pT', 'eta', 'matchedToGenPh',  'evt_rho', 'xs']

SaveDataFrameCSV,loadfromsaved=True,False #If loadfromsaved=True, dataframe stored in OutputDirName will be read

Classes,ClassColors = ['IsolatedSignal','NonIsolated'],['#377eb8', '#ff7f00']
#Remeber: For binary classification, first class of the Classes argument should be signal, otherwise, plots might not make sense.

processes=[
    {'Class':'IsolatedSignal','path':['/eos/user/a/atarabin/photonID/ID-Trainer/samples/hadded_low_newGenMatch.root', '/eos/user/a/atarabin/photonID/ID-Trainer/samples/hadded_high_newGenMatch.root'],
     #Can be a single root file, a list of root file, or even a folder but in a tuple format (folder,fileextension), like ('./samples','.root')
     'xsecwt': 'xs', #can be a number or a branch name, like 'weight' #Will go into training
     'selection':'(pT > 10) & ((abs(pho_SCEta) < 1.442) | (abs(pho_SCEta) > 1.566)) & (abs(eta) < 2.5) & (matchedToGenPh==1)', #selection for background
    },
    {'Class':'NonIsolated','path':['/eos/user/a/atarabin/photonID/ID-Trainer/samples/hadded_low_newGenMatch.root', '/eos/user/a/atarabin/photonID/ID-Trainer/samples/hadded_high_newGenMatch.root'],
     #Can be a single root file, a list of root file, or even a folder but in a tuple format (folder,fileextension), like ('./samples','.root')
     'xsecwt': 'xs', #can be a number or a branch name, like 'weight' #Will go into training
     'selection':'(pT > 10) & ((abs(pho_SCEta) < 1.442) | (abs(pho_SCEta) > 1.566)) & (abs(pho_SCEta) < 2.5)  & ((matchedToGenPh==0) | (matchedToGenPh==2))', #selection for background
    },
]

Tree = "ntuplizer/tree"

#MVAs to use as a list of dictionaries
MVAs = [
    #can add as many as you like: For MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc.
    #But keep XGB and DNN in the names (That is how the framework identifies which algo to run

    {"MVAtype":"XGB_1", #Keyword to identify MVA method.
     "Color":"green", #Plot color for MVA
     "Label":"XGB try1", # label can be anything (this is how you will identify them on plot legends)
     "features":["pho_SCRawE", "pho_R9Full5x5", "pho_SigmaIEtaIEtaFull5x5", "pho_SCEtaWidth", "pho_SCPhiWidth", "pho_SigmaIEtaIPhiFull5x5",
                 "pho_S4", "pho_ConeHoverE", "pho_EcalPFClusterIso", "pho_HcalPFClusterIso", "pho_trkSumPtHollowConeDR03",
                 "pho_trkSumPtSolidConeDR04", "pho_PFChIso", "pho_PFChWorstIso", "pho_SCEta", "evt_rho", "pho_ESEffSigmaRR",
                 "pho_PreShEbySCRawE"],
     "feature_bins":[100 for i in range(18)], #same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler", #Scaling for features before passing to the model training
     'UseGPU':False, #If you have a GPU card, you can turn on this option (CUDA 10.0, Compute Capability 3.5 required)
     "XGBGridSearch":{'min_child_weight': [5]} #All standard XGB parameters supported
    },
]


#------------------------------------------#------------------------------------------
######################################################################################################
######### Everything below this line is optinal ################################################

##############for 2D pt-eta reweighing
Reweighing = 'True' # This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples).
##############Even if this is 'False', xsec reweighting will always be carried to the training.
WhichClassToReweightTo="NonIsolated" #2D pt-eta spectrum of all other classs will be reweighted to this class
#------------------------------------------
ptbins = [10,15,18,20,23,26,30,33,37,40,44,48,53,60,70,85,250]
etabins = [-2.5,-2.2,-2.0,-1.8,-1.566,-1.442,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.442,1.566,1.8,2.0,2.2,2.5]
ptwtvar='pT'
etawtvar='pho_SCEta'
############# pt and eta bins of interest and branch names to read
############# (will be used for robustness studies and will also be used for 2D pt-eta reweighing) if the reweighing option is True

#------------------------------------------#Optional parameters below (Can be commented)
'''
#------------------------------------------
OverlayWP=['Fall17isoV2wp90','Fall17isoV2wp80'] # Working Points or flags to comapre to (should be booleans in the trees)
OverlayWPColors = ["black","purple"] #Colors on plots for WPs
#------------------------------------------
##############for 2D pt-eta reweighing
Reweighing = 'True' # This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples).
##############Even if this is 'False', xsec reweighting will always be carried to the training.
WhichClassToReweightTo="IsolatedSignal" #2D pt-eta spectrum of all other classs will be reweighted to this class
#------------------------------------------
ptbins = [5,10,30,40,50,80,100,5000]
etabins = [-1.6,-1.2,-0.8,-0.5,0.0,0.5,0.8,1.2,1.6]
ptwtvar='ele_pt'
etawtvar='scl_eta'
############# pt and eta bins of interest and branch names to read
############# (will be used for robustness studies and will also be used for 2D pt-eta reweighing) if the reweighing option is True
#------------------------------------------
SigEffWPs=["95%","98%"] # Example for 80% and 90% Signal Efficiency Working Points
############## To print thresholds of mva scores for corresponding signal efficiency
#------------------------------------------
RandomState=42
############### Choose the same number everytime for reproducibility
#------------------------------------------
MVAlogplot=False
############### If true, MVA outputs are plotted in log scale
#------------------------------------------
Multicore=False ### This is not very well tested!! Be careful!!
############### If True all CPU cores available are used XGB
#------------------------------------------
testsize=0.2
############### (0.2 means 20%) (How much data to use for testing)
#------------------------------------------
flatten=False
############## For NanoAOD and other un-flattened trees, you can switch on this option to flatten branches with variable length for each event
############## (Event level -> Object level).
############## You can't flatten branches which have different length for the same events. For example: It is not possible to flatten electron and muon branches both at the same time, since each event could have different electrons vs muons. Branches that have only one value for each event, line Nelectrons, can certainly be read along with unflattened branches.
#------------------------------------------
'''
