import os, sys
import torch
import numpy as np
from utils import word2sense, regression, listdir, image_to_tensor, Subject, cv_regression
from train import mean_condition_features
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import scipy.stats as stats
from collections import OrderedDict 
import matplotlib.pyplot as plt
import copy
from feature_extractors import AlexNetConv5, VGG16_post_conv5, VGG16_conv5
import seaborn as sns
import torch.nn as nn
import time
from tqdm import tqdm
from torchvision import datasets, models, transforms
import scipy.io as sio
import random

def feature_ablation(features, layer, idx):
    if layer == 'vgg16_conv5':
        features[0][idx][:][:] = 0
    return features

# feature extractors
vgg_conv5 = VGG16_conv5()
post_conv5 = VGG16_post_conv5()

imagenet_2012_synset = []
import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name
for i in range(0, 1000):
    imagenet_2012_synset.append(label_to_name(i))

imagenet_2012_table = {i: label_to_name(i) for i in range(1000)}
val_dir = './data/imagenet_ilsvrc'
val_gt = '/home/chan21/projects/semanticdimensionality/kh/object2vec_encoder_python/data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

with open(val_gt, 'r', encoding='utf-8') as txt:
    temp = txt.readlines() 
    
gt = [int(i.split('\n')[0]) for i in temp]
meta = sio.loadmat('/home/chan21/projects/semanticdimensionality/kh/object2vec_encoder_python/data/ILSVRC2012_devkit_t12/data/meta.mat')

ilsvrc2012_synset = []
for i in range(0, meta['synsets'].shape[0]):
    ilsvrc2012_synset.append(meta['synsets'][i][0][2][0])
    
vgg16 = models.vgg16(pretrained=True)
#vgg16.eval()

for param in vgg16.parameters():
    param.requires_grad = False
vgg16.classifier[6].weight.requires_grad = True

val_synset = []
ilsvrc2012_table = {}
for i in range(50000):
    target = gt[i]
    val_synset.append(meta['synsets'][target-1][0][2][0])
    ilsvrc2012_table[target] = meta['synsets'][target-1][0][2][0]
    
# ilsvrc2012_synset to imagenet_2012_table
map_table = {} #key: i (ilsvrc2012_synset), value: idx (imagenet_2012_table)
for i, item in ilsvrc2012_table.items():
    for idx, label in imagenet_2012_table.items(): 
        if label == item:
            #print(imagenet_2012_table[idx])
            #print(val_synset[i])
            map_table[i] = idx
            #print(imagenet_2012_table[idx])
            #print(imagenet_2012_table[map_table[i]])

# init random
random.seed = 0
randomlist = random.sample(range(0, 50000), 1000)


# 1. Baseline vgg16 classification
criterion = nn.CrossEntropyLoss()

total = len(randomlist)
total_loss = 0
n = 0
n1 = 0
logit = 0
for i in randomlist:
    #display(Image(filename=val_dir + '/ILSVRC2012_val_' + str(i+1).zfill(8) + '.JPEG', width=160, height=120))
    target = gt[i] # ilsvrc2012 target
    #print("Ground-truth: ", target, meta['synsets'][target-1][0][2][0])
    #print("Ground-truth (remapped to ImageNet2012): ", map_table[target], imagenet_2012_table[map_table[target]])
    output = vgg16(image_to_tensor(val_dir + '/ILSVRC2012_val_' + str(i+1).zfill(8) + '.JPEG').unsqueeze(0))
    #target_logit = output[0][map_table[target]]
    #logit += float(target_logit)
    loss = criterion(output, torch.tensor([map_table[target]]))
    total_loss += float(loss)
    values, indices = output.topk(5)
    #print("Top 5 prediction (remapped):", indices.numpy()[0], [imagenet_2012_table[index] for index in indices.numpy()[0]])
    gt_label = meta['synsets'][target-1][0][2][0]
    pre_labels = [imagenet_2012_table[index] for index in indices.numpy()[0]]
    if gt_label in pre_labels:
        n += 1
    if gt_label == pre_labels[0]:
        n1 += 1

top5_acc = n/total
top1_acc = n1/total
#logit = logit/total
avg_loss = total_loss/total

print(top5_acc, top1_acc, avg_loss)
baseline_vgg16_conv5 = [top5_acc, top1_acc, avg_loss]
np.save('classify_baseline_vgg16_conv5.npy', baseline_vgg16_conv5)

# 2. Ablated vgg16 classification

ablation_vgg16_conv5_dict = {}

for fmap in range(512):
    total = len(randomlist)
    total_loss = 0
    n = 0
    n1 = 0
    logit = 0
    for i in randomlist:
        #display(Image(filename=val_dir + '/ILSVRC2012_val_' + str(i+1).zfill(8) + '.JPEG', width=160, height=120))
        target = gt[i] # ilsvrc2012 target
        #print("Ground-truth: ", target, meta['synsets'][target-1][0][2][0])
        #print("Ground-truth (remapped to ImageNet2012): ", map_table[target], imagenet_2012_table[map_table[target]])

        conv5 = vgg_conv5(image_to_tensor(val_dir + '/ILSVRC2012_val_' + str(i+1).zfill(8) + '.JPEG').unsqueeze(0))
        conv5[0][fmap][:][:] = 0
        #features[0][idx][:][:] = feature_ablation(conv5, 'vgg16_conv5', fmap)
        output = post_conv5(conv5)
        # output = vgg16()

        #target_logit = output[0][map_table[target]]
        #logit += float(target_logit)
        loss = criterion(output, torch.tensor([map_table[target]]))
        total_loss += float(loss)
        values, indices = output.topk(5)
        #print("Top 5 prediction (remapped):", indices.numpy()[0], [imagenet_2012_table[index] for index in indices.numpy()[0]])
        gt_label = meta['synsets'][target-1][0][2][0]
        pre_labels = [imagenet_2012_table[index] for index in indices.numpy()[0]]
        if gt_label in pre_labels:
            n += 1
        if gt_label == pre_labels[0]:
            n1 += 1
            
        #del img
        del conv5
        #del ablated_conv5
        del output
        
    top5_acc = n/total
    top1_acc = n1/total
    #logit = logit/total
    avg_loss = total_loss/total

    print(top5_acc, top1_acc, avg_loss)
    ablation_vgg16_conv5_dict[fmap] = [top5_acc, top1_acc, avg_loss]
    
    

np.save('classify_ablation_vgg16_conv5.npy', ablation_vgg16_conv5_dict) 
