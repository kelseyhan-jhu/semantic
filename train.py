import os
from argparse import ArgumentParser
import torch
from Encoder import Encoder
from utils import listdir, cv_regression

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--rois', nargs='+', default=['LOC'], type=str, help='ROIs to fit, separated by spaces.' 
                        'Options include: EVC, LOC, PFS, OPA, PPA, RSC, FFA, OFA, STS, EBA')
    parser.add_argument('--subject_number', default=1, type=int, help='Subject number to train encoder for', choices=[1, 2, 3, 4])
    parser.add_argument('--train_data', default='object2vec')
    parser.add_argument('--model', default='alexnet')
    parser.add_argument('--pretrained', default='pretrained')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--l2', default=0, type=float, help='L2 regularization weight')
    parser.add_argument('--maxpool', default='', help='global maxpooling on features', choices=['', 'maxpool'])
    args = parser.parse_args()

    # load features
    layers = {1: 'conv1', 2: 'conv2', 3: 'conv3', 4: 'conv4', 5: 'conv5', 6: 'fc6', 7: 'fc7'}
    layer = layers[args.layer]
    feature_path = os.path.join("./processed/feature", args.train_data, args.model)
    conditions = listdir(feature_path)
    condition_features = {}
    for c in conditions:
        features = listdir(c)
        c_name = c.split('/')[-1]
        if args.maxpool == 'maxpool':
            if (layer != 'fc6') and (layer != 'fc7'):
                features = torch.stack([torch.load(f)[layer] for f in features])
                features, _ = torch.max(features.view(features.size(0), features.size(1), features.size(2), features.size(3)*features.size(4)).squeeze(), -1) # global maxpooling
                #features = torch.stack([torch.tensor(torch.load(f)[layer].numpy().max(axis=-1).max(axis=-1)).flatten() for f in features])
        else:
            features = torch.stack([torch.load(f)[layer].flatten() for f in features])

        features = features.mean(dim=0)
        condition_features[c_name] = features
        
    # fmri prep
    for r in args.rois:
        encoder = Encoder(args.subject_number, args.rois)
            
    # train
    weights, r = cv_regression(condition_features, encoder, l2=args.l2)
    
    
    name = args.model + str(args.layer) + '_' + ''.join(args.rois) + '_' + args.maxpool
    print(name, r)
    
    if not os.path.exists(os.path.join('processed/saved_encoder/', str(args.subject_number))):
        os.mkdir(os.path.join('processed/saved_encoder/', str(args.subject_number)))
    torch.save(weights, os.path.join('./processed/saved_encoder', str(args.subject_number), name + '.pth'))