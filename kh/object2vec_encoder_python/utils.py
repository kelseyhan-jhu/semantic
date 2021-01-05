import os
from PIL import Image
import numpy as np
import scipy.io
import torch
from torchvision.transforms import functional as tr
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from collections import OrderedDict 
from scipy import stats

# ImageNet mean and standard deviation. All images
# passed to a PyTorch pre-trained model (e.g. AlexNet) must be
# normalized by these quantities, because that is how they were trained.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def p2r(p, n):
    t = stats.t.ppf(1-p, n-2);
    r = (t**2/((t**2)+(n-2))) ** 0.5;
    return r

def stack_features(path, fmri=True):
    if fmri is True:
        conditions = sorted(listdir(path))
        condition_voxels = {}
        for condition in conditions:
            #print(condition)
            #print(listdir(condition))
            file_name = (listdir(condition)[0].split('/'))[-1]
            #print(file_name)
            file_path = os.path.join(condition, file_name)
            #print(file_path)
            condition_voxels[condition] = np.load(file_path) 
        features_stacked = np.stack([condition_voxel for condition, condition_voxel in OrderedDict(condition_voxels).items()])
        #print(np.shape(pred_voxels)) # should be 1470, 200
    elif fmri is False: 
        condition_features = torch.load(path)
        features_stacked = np.stack([condition_feature for condition, condition_feature in OrderedDict(condition_features).items()])
        #print(stacked_features.shape)
    return features_stacked

def cv_regression_w2s(features, w2s, fit=None, k=9, l2 = 0.0, pc=None):
    if pc is not None:
        pca = PCA(n_components=pc)
    kf = KFold(n_splits = k)
    rs = []
    for train_index, test_index in kf.split(features):
        train_features = features[train_index,]
        test_features =  features[test_index,]
        
        if pc is not None:
            pca.fit(train_features)   
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
        train_embeddings = np.stack([embedding for i, embedding in enumerate(w2s.values()) if i in train_index])
        
        #print(np.nanmean(train_embeddings.astype('float64')))
        
        #imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

        #imp_mean.fit(train_embeddings.astype('float64'))
        #train_embeddings = imp_mean.transform(train_embeddings.astype('float64'))

        #print(np.nanmean(train_embeddings))
        
        test_embeddings = np.stack([embedding for i, embedding in enumerate(w2s.values()) if i in test_index])
        #imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        #imp_mean.fit(test_embeddings.astype('float64'))
        #test_embeddings = imp_mean.transform(test_embeddings.astype('float64'))
        
        weights, r = regression(train_features, train_embeddings, test_features, test_embeddings, l2=l2)
        rs.append(r)
        
    rs = np.array(rs)
    
    mean_r = np.nanmean(rs, axis=0) # mean across k folds 

    return weights, mean_r

def word2sense(embedding_file):
    f = open(embedding_file, 'r', encoding='utf-8')
    temp = f.readlines()
    wordlist = [line.split(',')[0] for idx, line in enumerate(temp) if idx != 0] # create word list
    f.close()
    word2sense = {new_list: [] for new_list in wordlist} # create (word: embedding) dictionary
    for i, line in enumerate(temp):
        if i == 0:
            continue
        embedding = line.split(',')
        embedding.remove('\n')
        embedding_float = ([float(j) for j in embedding[1:]])
        word2sense[embedding[0]] = embedding_float
    word2sense = OrderedDict(word2sense)
    return wordlist, word2sense

def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if (f != '.DS_Store' and f != '._.DS_Store' and f != '.ipynb_checkpoints')]
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image, resolution=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:     # if not square image, crop the long side's edges
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image


def tensor_to_image(image, do_imagenet_unnorm=True):
    if do_imagenet_unnorm:
        image = imagenet_unnorm(image)
    image = tr.to_pil_image(image)
    return image


def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image


def imagenet_unnorm(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    image = image.cpu()
    image = image * std + mean
    return image


class Subject:

    def __init__(self, subject_num, rois):
        roistack = scipy.io.loadmat('data/fmri/subj{:03}'.format(subject_num) +
                                    '/roistack.mat')['roistack']
        self.roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        self.conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0]
        roi_masks = {roi: roi_indices == (i + 1) for i, roi in enumerate(self.roi_names)}
        voxels = roistack['betas'][0, 0]
        self.condition_voxels = {cond: np.concatenate([voxels[i][roi_masks[r]] for r in rois])
                                 for i, cond in enumerate(self.conditions)}
        self.n_voxels = np.sum([roi_masks[r] for r in rois])

        sets = scipy.io.loadmat('data/fmri/subj{:03}'.format(subject_num) +
                                   '/sets.mat')['sets']
        self.cv_sets = [[cond[0] for cond in s[:, 0]] for s in sets[0, :]]

def cv_regression(condition_features, subject, l2=0.0):
    # Get cross-validated mean test set correlation
    rs = []
    
    pca = PCA(n_components=10)

    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]
        
        train_features = np.stack([condition_features[c] for c in train_conditions])
        test_features = np.stack([condition_features[c] for c in test_conditions])
        train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])
        #_, r = L1_regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
        _, r = regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
        rs.append(r)
        
        pca.fit(train_voxels)   
        print(pca.explained_variance_ratio_)

    mean_r = np.mean(rs)

    # Train on all of the data
    train_conditions = subject.conditions
    train_features = np.stack([condition_features[c] for c in train_conditions])
    train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
    weights = regression(train_features, train_voxels, None, None, l2=l2, validate=False)
    #_, r = L1_regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
    
    return weights, mean_r


# def cv_regression_w2s(condition_Things, w2s, l2=0.0):
#     # Get cross-validated mean test set correlation
#     rs = []
    
#     for test_conditions in subject.cv_sets:
#         train_conditions = [c for c in subject.conditions if c not in test_conditions]
#         train_features = np.stack([condition_features[c] for c in train_conditions])
#         test_features = np.stack([condition_features[c] for c in test_conditions])
#         train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
#         test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])
#         #_, r = L1_regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
#         _, r = regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
#         rs.append(r)
#     mean_r = np.mean(rs)

#     # Train on all of the data
#     train_conditions = subject.conditions
#     train_features = np.stack([condition_features[c] for c in train_conditions])
#     train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
#     weights = regression(train_features, train_voxels, None, None, l2=l2, validate=False)
#     #_, r = L1_regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
    
#     return weights, mean_r


def regression(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    r_ = []
    if validate:
        y_pred = regr.predict(x_test)
        y_pred = y_pred.transpose()
        y_test = y_test.transpose()
        for (y_t, y_p) in zip(y_test, y_pred):
            r = correlation(y_t, y_p)
            r_.append(r)
            #print(r)
        return weights, r_
    else:
        return weights
    

def regression_(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    
    r_ = {}
    if validate:
        y_pred = regr.predict(x_test)
        # y_pred.shape (9, 195) (stim-1, voxel size)
        for (y_t, y_p) in zip(y_test, y_pred):
            r = correlation(y_t, y_p)
            r_.append(r)
            print(r)
        return weights, r_, y_pred
    else:
        return weights

    
def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean()
    return r
