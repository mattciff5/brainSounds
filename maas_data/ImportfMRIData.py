# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:54:12 2026

@author: E.Formisano
"""
import h5py
import numpy as np

#=========================
#  Load fMRI data (train and test)
# ============================


pathf ="E:/fMRIData_betas"

name_sounds=np.load(pathf+"/SoundNames.npy")

fname_train = pathf +  "/pSTG/CV1/BetasTrain_KV.mat"  # change this for other CVs and Subjects
fname_test  = pathf + "/pSTG/CV1/BetasTest_KV.mat"

with h5py.File(fname_train, 'r') as f:
    fMRI_train = np.array(f["BetasTrain"])

with h5py.File(fname_test, 'r') as f:
    fMRI_test = np.array(f["BetasTest"])


print("fMRI_train shape:", fMRI_train.shape)
print("fMRI_test shape:", fMRI_test.shape)

# Zscore ->  per voxel
#train_mean_fmri = fMRI_train.mean(axis=1, keepdims=True)
#train_std_fmri  = fMRI_train.std(axis=1, keepdims=True)

#fMRI_train = (fMRI_train - train_mean_fmri) / train_std_fmri
#fMRI_test  = (fMRI_test  - train_mean_fmri) / train_std_fmri


# ============================
#  Load the index of sounds in train e test 
# ============================

# ----- trainSounds -----
with h5py.File(pathf+"/pSTG/CV1/trainSounds.mat", "r") as f:
    ts = np.array(f["trainSounds"])
    ts = ts.squeeze()

# ----- testSounds -----
with h5py.File(pathf+"/pSTG/CV1/testSounds.mat", "r") as f:
    vs = np.array(f["testSounds"])
    vs = vs.squeeze()

print("trainSounds shape:", ts.shape)
print("testSounds  shape:", vs.shape)

# MATLAB usa indici da 1 → converto a 0-based per Python
idx_train = ts.astype(int) - 1
idx_test  = vs.astype(int) - 1

