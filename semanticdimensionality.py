import os, sys
from glob import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import p2r, listdir, image_to_tensor, regression
from tqdm import tqdm
import warnings; warnings.simplefilter('ignore')
from collections import OrderedDict 
import string
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold



def load_embedding(embedding='w2s'):
    if embedding == 'w2s':
        w_file = os.path.join('./data/embedding/', 'ThingsWrd2Sns' + '.txt')
        _, embed = word2sense(w_file)
        nDim = 2250
        nCategory = 1470
        r_threshold = p2r(0.05/nDim, nCategory)
        return embed, nDim, r_threshold
    
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
    
def cv_regression_w(features, wordembedding, fit=None, k=9, l2=0.0, pc_fmri=None, pc_embedding=None):
    if pc_fmri is not None:
        pca_fmri = PCA(n_components=pc_fmri)
    if pc_embedding is not None:
        pca_embedding = PCA(n_components=int(pc_embedding))
    kf = KFold(n_splits = k)
    rs = []
    for train_index, test_index in kf.split(features):
        train_features = features[train_index,]
        test_features =  features[test_index,]
        if pc_fmri is not None:
            pca_fmri.fit(train_features)   
            train_features = pca_fmri.transform(train_features)
            test_features = pca_fmri.transform(test_features)
            
        train_embeddings = np.stack([embedding for i, embedding in enumerate(wordembedding.values()) if i in train_index])
        test_embeddings = np.stack([embedding for i, embedding in enumerate(wordembedding.values()) if i in test_index])
        if pc_embedding is not None:
            pca_embedding.fit(train_embeddings)
            train_embeddings = pca_embedding.transform(train_embeddings)
            test_embeddings = pca_embedding.transform(test_embeddings)

        weights, r = regression(train_features, train_embeddings, test_features, test_embeddings, l2=l2)
        rs.append(r)
    rs = np.array(rs)
    mean_r = np.nanmean(rs, axis=0) # mean across k folds
    
    return weights, mean_r


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--predicted_data", default='things')
    parser.add_argument('--name', required=True, type=str,
                        help='predicted responses for which to compute semantic dimensionality')
    parser.add_argument('--embedding', default='w2s')
    parser.add_argument('--subject',  nargs='+', default=['1', '2', '3', '4'], type=str, help='subjects to concat, separated by spaces.')
    parser.add_argument('--random', default='random', help='if set to random, include the respective random encoder', choices=['', 'random'])
    parser.add_argument('--pc_embedding', type=str)
    parser.add_argument('--output_file', required=True, type=str)
    args = parser.parse_args()
    
    # Setup embedding
    embed, nDim, r_threshold = load_embedding(embedding=args.embedding)
    
    # parse name (e.g. alexnet5_LOC_maxpool)
    enc_name = args.name.split('_')
    model = enc_name[0][0:-1] 
    layer = int(enc_name[0][-1])
    roi = enc_name[1]
    maxpool = enc_name[2]

    roi_r = []
    random_r = []
    sub_voxel_regressor = {}

    for subj in args.subject:
        path = os.path.join('processed/predicted/', args.predicted_data, subj, args.name + '_' + args.random)
        conditions = sorted(embed.keys())
        condition_voxels = {}
        for condition in conditions:
            file_name = listdir((os.path.join(path, condition)))[0]
            features = np.load(file_name)
            condition_voxels[condition] = np.mean(features, axis=0)
        voxel_regressor = np.stack([condition_voxel for condition, condition_voxel in OrderedDict(condition_voxels).items()])
        sub_voxel_regressor[subj] = voxel_regressor
    
    all_voxel_regressor = np.hstack([sub_voxel_regressor[s] for s in args.subject])

    for pc in tqdm(range(10, 80, 10), total=7, position=0, leave=True):
        _, voxel_mean_r = cv_regression_w(all_voxel_regressor, embed, fit=None, k=9, l2=0.0, pc_fmri=pc, pc_embedding=args.pc_embedding)
        roi_r.append(voxel_mean_r)
    
    roi_dim = []
    roi_max = []
    roi_mean = []
    roi_med = []       
    
    for dim_pred in roi_r:
        dim_pred = np.array(dim_pred)
        dim_pred_score = np.sum(dim_pred > r_threshold)
        dim_max = np.nanmax(dim_pred)
        dim_mean = np.nanmean(np.array(dim_pred))
        dim_med = np.nanmedian(np.array(dim_pred))
        roi_dim.append(dim_pred_score)
        roi_max.append(dim_max)
        roi_mean.append(dim_mean)
        roi_med.append(dim_med)  
        
    print(roi_dim)
    
    with open(args.output_file + '.txt', 'a') as f:
        print(args.predicted_data + '/' + ''.join(args.subject) + '/' + args.name + '_' + args.random, file=f)
        print(roi_dim, file=f) #roi_max, roi_mean, roi_med