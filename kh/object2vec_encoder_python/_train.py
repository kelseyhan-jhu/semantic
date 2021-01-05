from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch
from feature_extractors import AlexNetFC6
from Encoder import Encoder
from utils import listdir, image_to_tensor, Subject, cv_regression


def mean_condition_features(model, resolution):
    print('Extracting stimuli features')
    conditions = listdir('data/stimuli')
    condition_features = {}
    for c in tqdm(conditions):
        c_name = c.split('/')[-1]
        stimuli = listdir(c)
        stimuli = [image_to_tensor(s, resolution=resolution) for s in stimuli]
        stimuli = torch.stack(stimuli)
        #if torch.cuda.is_available():
        #    stimuli = stimuli.cuda()
        with torch.no_grad():
            feats = model(stimuli).mean(dim=0).cpu().numpy()
        condition_features[c_name] = feats
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Train encoder using object2vec study data')
    parser.add_argument('--name', required=True, type=str,
                        help='Name to save trained encoder with (will be saved as saved_encoders/[name].pth)')
    parser.add_argument('--rois', nargs='+', default=['LOC'], type=str,
                        help='ROIs to fit, separated by spaces. '
                             'Options include: EVC, LOC, PFS, OPA, PPA, RSC, FFA, OFA, STS, EBA')
    parser.add_argument('--subject_number', default=1, type=int, help='Subject number to train encoder for',
                        choices=[1, 2, 3, 4])
    parser.add_argument('--resolution', default=256, type=int, help='Resolution at which to resize stimuli')
    parser.add_argument('--feature_extractor', default='alexnetfc6', type=str, help='Feature extraction model')
    parser.add_argument('--l2', default=0, type=float, help='L2 regularization weight')
    args = parser.parse_args()

    if args.feature_extractor == 'alexnetfc6':
        feat_extractor = AlexNetFC6()
    else:
        raise ValueError('Unimplemented feature extractor: {}'.format(args.feature_extractor))

    subject = Subject(args.subject_number, args.rois)
    condition_features = mean_condition_features(feat_extractor, args.resolution)

    weights, r = cv_regression(condition_features, subject, l2=args.l2)
    print('Mean test set correlation (r) over cross-validated folds: {:.4g}'.format(r))

    encoder = Encoder(feat_extractor, weights)
    torch.save(encoder, os.path.join('saved_encoders', args.name + '.pth'))
