import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
import copy

features = torch.load('features_conv5.pth')
features_arr = np.stack([f for f in features.values()])

rdms = {n: np.zeros((len(features), len(features))) for n in np.arange(0, 256, 10)}
rdms[0] = np.load("UnitRemoval/rsa_baseline.npy")

# Load visual -> semantic feature ranking
acc = np.load('./UnitRemoval/r_vis2sem.npy')

features_reshape = copy.deepcopy(features)
for item in features_reshape:
    features_reshape[item] = np.reshape(features_reshape[item], (256, 7, 7))

for n in tqdm(range(0, 256, 10)):
    if n == 0:
        continue
    features_temp = copy.deepcopy(features_reshape)
    maxidx = np.argsort(np.array(acc))[:n]
    for idx in maxidx:
        for item in features_temp:
            features_temp[item][idx][:][:] = 0
    features_arr = np.stack([np.reshape(feature, (12544,)) for feature in features_temp.values()])
    for i, c in enumerate(features_arr):
        for j, d in enumerate(features_arr): 
            r, p = stats.pearsonr(c, d) # computing pairwise correlations of betas
            rdms[n][i, j] = r
        if i % 100 == 0:
            print(i, r)

np.save("rsa_2a", rdms)

# Repeat for 2-b. (ranking by fMRI response prediction)
rdms_b = {n: np.zeros((len(features), len(features))) for n in np.arange(0, 256, 10)}
rdms_b[0] = np.load("UnitRemoval/rsa_baseline.npy")

# Load visual -> semantic feature ranking
acc = np.load('./UnitRemoval/r_vis2fmri.npy')

for n in tqdm(range(0, 256, 10)):
    if n == 0:
        continue
    features_temp = copy.deepcopy(features_reshape)
    maxidx = np.argsort(np.array(acc))[:n]
    for idx in maxidx:
        for item in features_temp:
            features_temp[item][idx][:][:] = 0
    features_arr = np.stack([np.reshape(feature, (12544,)) for feature in features_temp.values()])
    for i, c in enumerate(features_arr):
        for j, d in enumerate(features_arr): 
            r, p = stats.pearsonr(c, d) # computing pairwise correlations of betas
            rdms_b[n][i, j] = r
        if i % 100 == 0:
            print(i, r)
            
np.save("rsa_2b", rdms_b)
