from argparse import ArgumentParser
import os
import shutil
from tqdm import tqdm
import numpy as np
import torch
from utils import listdir, image_to_tensor


def get_predictions(model, resolution, stimuli_folder, batch_size=32):
    print('Getting model predictions')
    stimuli = listdir(stimuli_folder)
    predictions = {}
    #for i in tqdm(range(0, len(stimuli), batch_size)):
    for i in range(0, len(stimuli), batch_size):
        batch_stimuli = stimuli[i:i + batch_size]
        batch_tensors = torch.stack([image_to_tensor(s, resolution=resolution) for s in batch_stimuli])
        #if torch.cuda.is_available():
        #    batch_tensors = batch_tensors.cuda()
        with torch.no_grad():
            batch_preds = model(batch_tensors).cpu().numpy()
        for stimulus_path, pred in zip(batch_stimuli, batch_preds):
            stimulus_name = stimulus_path.split('/')[-1]
            predictions[stimulus_name] = pred
    return predictions

if __name__ == '__main__':
    parser = ArgumentParser(description='Run trained encoder on stimulus folder and save numpy results')
    parser.add_argument('--name', required=True, type=str,
                        help='Name of saved trained encoder (will be loaded from saved_encoders/[name].pth)')
    parser.add_argument('--stimuli_folder', required=True, type=str,
                        help='Path to directory with stimuli for which to get encoder predictions')
    parser.add_argument('--save_folder', required=True, type=str,
                        help='Path to directory to save predictions (will be deleted if already exists)')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution at which to resize stimuli')
    args = parser.parse_args()

    encoder = torch.load(os.path.join('saved_encoders', args.name + '.pth'),
                         map_location=lambda storage, loc: storage)

    category = listdir(args.stimuli_folder)
    category_stimuli = {}
    
    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    for c in tqdm(category):
        c_name = c.split('/')[-1]
        predictions = get_predictions(encoder, args.resolution, c)
        
        os.mkdir(args.save_folder + '/' + c_name)
        for stimulus_name, pred in predictions.items():
            np.save(os.path.join(args.save_folder, c_name, stimulus_name), pred)
