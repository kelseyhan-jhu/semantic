from argparse import ArgumentParser
import os
import shutil
from tqdm import tqdm
import numpy as np
import torch
from utils import listdir, image_to_tensor
from utils import word2sense

def get_predictions(model, resolution, stimuli_folder, batch_size=32):
    #print('Getting model predictions')
    stimuli = listdir(stimuli_folder)
    predictions = {}
    stimuli = [image_to_tensor(s, resolution=resolution) for s in stimuli]
    stimuli = torch.stack(stimuli).to(device)
    with torch.no_grad():
        preds = model(stimuli).mean(dim=0).cpu().numpy() 
    #for stimulus_path, pred in zip(stimuli, preds):
    stimulus_name = stimuli_folder.split('/')[-1]
    predictions[stimulus_name] = preds
        #print(predictions[stimulus_name].shape) #(200,0)
    return predictions
    #for i in tqdm(range(0, len(stimuli), batch_size)):
#     for i in range(0, len(stimuli), batch_size):
#         batch_stimuli = stimuli[i:i + batch_size]
#         batch_tensors = torch.stack([image_to_tensor(s, resolution=resolution) for s in batch_stimuli])
        
#         #if torch.cuda.is_available():
#         #    batch_tensors = batch_tensors.cuda()
#         with torch.no_grad():
#             batch_preds = model(batch_tensors).cpu().numpy()
#         for stimulus_path, pred in zip(batch_stimuli, batch_preds):
#             stimulus_name = stimulus_path.split('/')[-1]
#             predictions[stimulus_name] = pred
#             #print(predictions[stimulus_name].shape) #(200,0)
#     return predictions

def get_predictions_ww(features, weight, c):
    predictions = {}
    c_name = c.split('/')[-1]
    stimuli = listdir(c)
    stimuli = [image_to_tensor(s, resolution=resolution) for s in stimuli]
    stimuli = torch.stack(stimuli) # stacked in 10
    #if torch.cuda.is_available():
    #    stimuli = stimuli.cuda()
    with torch.no_grad():
        feats = model(stimuli).mean(dim=0).cpu().numpy()
    predictions[c_name] = feats
    
    preds = np.matmul(f, weight)
    print(preds.shape)
    stimulus_name = stimuli_folder.split('/')[-1]
    predictions[stimulus_name] = preds

    return predictions

if __name__ == '__main__':
    parser = ArgumentParser(description='Run trained encoder on stimulus folder and save numpy results')
    parser.add_argument('--name', required=True, type=str,
                        help='Name of saved trained encoder/weight (will be loaded from saved_encoders/[name].pth)')
    parser.add_argument('--feature_extractor', required=True, default='alexnetconv5', type=str, help='Feature extraction model')
    parser.add_argument('--stimuli_folder', required=True, type=str,
                        help='Path to directory with stimuli for which to get encoder predictions')
    parser.add_argument('--save_folder', required=True, type=str,
                        help='Path to directory to save predictions (will be deleted if already exists)')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution at which to resize stimuli')
    parser.add_argument('--trained_random', default='trained', help='Predict with trained or random encoder')
    args = parser.parse_args()
    
    #encoder = torch.load(os.path.join('saved_encoders', args.name + '.pth'),
    #                     map_location=lambda storage, loc: storage).to(device)

    category = listdir(args.stimuli_folder)     
    category_stimuli = {}
    words, _ = word2sense("ThingsWrd2Sns.txt")
    
    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    if args.feature_extractor == 'alexnetfc6':
        features = torch.load('features_fc6.pth')
    elif args.feature_extractor == 'alexnetconv5':
        features = torch.load('features_conv5.pth')
    elif args.feature_extractor == 'alexnetconv1':
        features = torch.load('features_conv1.pth')
    elif args.feature_extractor == 'vgg16conv5':
        features = torch.load('features_vgg16_conv5.pth')
    else:
        raise ValueError('Unimplemented feature extractor: {}'.format(args.feature_extractor))
    
    if args.trained_random == 'trained':
        weight = np.load(os.path.join('saved_weights/vgg', args.name) + '.npy')
    elif args.trained_random == 'random':
        scaffold = np.load(os.path.join('saved_weights/vgg', args.name) + '.npy')
#         fmri_mean = np.mean(scaffold, axis=0)
#         fmri_cov = np.cov(scaffold, rowvar=False)
#         #weight = np.random.multivariate_normal(fmri_mean, fmri_cov, (scaffold.shape[0]))
#         weight = np.random.multivariate_normal(fmri_mean, fmri_cov, (scaffold.shape[0]))
#         #weight = np.random.rand(scaffold.shape[0], scaffold.shape[1])
        fmri_mean2 = np.mean(scaffold, axis=1)
        fmri_cov2 = np.cov(scaffold, rowvar=True)
        weight2 = np.random.multivariate_normal(fmri_mean2, fmri_cov2, (scaffold.shape[1]))
        weight = weight2.T
        print(weight.shape)
    elif args.trained_random == 'permuted':
        scaffold = np.load(os.path.join('saved_weights', args.name) + '.npy')
        permutation = np.random.choice(scaffold.shape[1], scaffold.shape[1], replace=False)
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        weight = scaffold[:, idx]
    else:
        raise ValueError('Cannot have values other than trained or random: {}'.format(args.trained_random))
            
    for c in tqdm(category, total=len(category), position=0, leave=True):
        c_name = c.split('/')[-1]
        #predictions = get_predictions(encoder, args.resolution, c)
        if c_name in words:
            key_name = "/home/chan21/projects/semanticdimensionality/kh/object2vec_encoder_python/images/" + c_name
            #predictions = get_predictions(encoder, args.resolution, c)
            feature = features[key_name]
            #predictions = get_predictions_ww(weight, feature, c)
            pred = np.matmul(feature, weight.T)
            os.mkdir(args.save_folder + '/' + c_name)
            np.save(os.path.join(args.save_folder, c_name, c_name), pred)
            #for stimulus_name, pred in predictions.items():
            #    np.save(os.path.join(args.save_folder, c_name, stimulus_name), pred)
