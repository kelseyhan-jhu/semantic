from argparse import ArgumentParser
import os
import shutil
from tqdm import tqdm
import numpy as np
import torch
from utils import listdir, image_to_tensor

if __name__ == '__main__':
    parser = ArgumentParser(description='Run trained encoder on stimulus folder and save numpy results')
    parser.add_argument('--name', required=True, type=str,
                        help='Name of saved trained encoder/weight (will be loaded from saved_encoders/[name].pth)')
    parser.add_argument('--stimuli_folder', required=True, type=str,
                        help='Path to directory with stimuli for which to get encoder predictions')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution at which to resize stimuli')
    parser.add_argument('--random', default='', help='Predict with trained or random encoder', choices=['', 'random'])
    args = parser.parse_args()
    
    
    shutil.rmtree(os.path.join('processed/predicted/', args.stimuli_folder, args.name), ignore_errors=True)
    os.mkdir(os.path.join('processed/predicted/', args.stimuli_folder, args.name))
    
    # parse encoder name (e.g. 1_alexnet5_LOC_maxpool)
    enc_name = args.name.split('_')
    subject = enc_name[0]
    model = enc_name[1][0:-1] 
    layer = int(enc_name[1][-1])
    roi = enc_name[2]
    maxpool = enc_name[3]
    
    print(subject, model, layer, roi, maxpool)
    
    if args.random == '':
        weight = torch.load(os.path.join('processed/saved_encoder/', args.name) + '.pth')
    elif args.random == 'random':
        temp = torch.load(os.path.join('processed/saved_encoder/', args.name) + '.pth')
        weight_p = np.random.choice(temp.shape[1], temp.shape[1], replace=False)
        idx = np.empty_like(weight_p)
        idx[weight_p] = np.arange(len(weight_p))
        weight = temp[:, idx]
    else:
        raise ValueError('Cannot have values other than empty or random: {}'.format(args.random))
    
    
    # load features
    layers = {1: 'conv1', 2: 'conv2', 3: 'conv3', 4: 'conv4', 5: 'conv5', 6: 'fc6', 7: 'fc7'}
    layer = layers[layer]

    feature_path = os.path.join("./processed/feature", args.stimuli_folder, model)
    conditions = listdir(feature_path)
    condition_features = {}
    
    for c in tqdm(conditions, total=len(conditions), position=0, leave=True):
        features = listdir(c)
        c_name = c.split('/')[-1]
        if maxpool == 'maxpool' and layer != 'fc6' and layer != 'fc7':
            features = torch.stack([torch.load(f)[layer] for f in features])
            features, _ = torch.max(features.view(features.size(0), features.size(1), features.size(2), features.size(3)*features.size(4)).squeeze(), -1) # global maxpooling
            #features = torch.stack([torch.tensor(torch.load(f)[layer].numpy().max(axis=-1).max(axis=-1)).flatten() for f in features])
        else:
            features = torch.stack([torch.load(f)[layer].flatten() for f in features])

        pred = np.matmul(features, weight.T)
        os.mkdir(os.path.join('processed/predicted', args.stimuli_folder, args.name, c_name))
        np.save(os.path.join('processed/predicted/', args.stimuli_folder, args.name, c_name, c_name), pred)
        #for stimulus_name, pred in predictions.items():
        #    np.save(os.path.join(args.save_folder, c_name, stimulus_name), pred)
