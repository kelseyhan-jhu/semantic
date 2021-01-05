import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
from utils import listdir, image_to_tensor

from feature_extractors import AlexNetConv5, VGG16
from tqdm import tqdm 
#feat_extractor = AlexNetConv5()
feat_extractor = VGG16()
              
# images = [listdir(d) for d in listdir('images')]
# images = sum(images, [])
# image_features = {}

# for image in tqdm(images):
#     im_tensor = image_to_tensor(image, resolution=256)
#     im_tensor = torch.reshape(im_tensor, (1, 3, 256, 256))
#     im = image.split('/')[-2] + '/' + image.split('/')[-1]
#     with torch.no_grad():
#         feats = feat_extractor(im_tensor).mean(dim=0).cpu().numpy()
#     image_features[im] = feats

# #np.save("features_conv5_all.pth", image_features)
# torch.save(image_features, "features_vgg16_conv5_all.pth")
               
images = [listdir(d) for d in listdir('data/stimuli')]
images = sum(images, [])
image_features = {}

for image in tqdm(images):
    im_tensor = image_to_tensor(image, resolution=256)
    im_tensor = torch.reshape(im_tensor, (1, 3, 256, 256))
    im = image.split('/')[-2] + '/' + image.split('/')[-1]
    with torch.no_grad():
        feats = feat_extractor(im_tensor).mean(dim=0).cpu().numpy()
    image_features[im] = feats

#np.save("features_conv5_all_o2v.pth", image_features)
torch.save(image_features, "features_vgg16_conv5_all_o2v.pth")

