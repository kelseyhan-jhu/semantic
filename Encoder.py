import os, sys
import scipy.io
import numpy as np

class Encoder:
    '''in silico model with image features as input and fMRI beta values as output
    '''
    def __init__(self, subject, rois):    
        self.subject = subject
        self.rois = rois
        roistack = scipy.io.loadmat('data/fmri/object2vec/subj{:03}'.format(self.subject) +
                                        '/roistack.mat')['roistack']
        self.roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        self.conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0]
        roi_masks = {r: roi_indices == (i + 1) for i, r in enumerate(self.roi_names)}
        voxels = roistack['betas'][0, 0]
        self.condition_voxels = {cond: np.concatenate([voxels[i][roi_masks[r]] for r in rois])
                                 for i, cond in enumerate(self.conditions)}
        self.n_voxels = np.sum([roi_masks[r] for r in rois])

        sets = scipy.io.loadmat('data/fmri/object2vec/subj{:03}'.format(self.subject) +
                                   '/sets.mat')['sets']
        self.cv_sets = [[cond[0] for cond in s[:, 0]] for s in sets[0, :]]
        
        