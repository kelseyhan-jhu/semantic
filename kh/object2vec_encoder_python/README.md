# Object2Vec Encoder Python
Python repository for training an encoder on the Object2Vec fMRI dataset and running it on new stimuli using PyTorch.

## Required packages
* PyTorch (and Torchvision)
* Numpy
* PILLOW
* scipy
* sklearn
* tqdm

## Training
To train the encoder, run the `train.py` script (see script for arguments). You can extend this file
and the ones that it uses to add new feature extractors other than AlexNet.

**Note: The repository only comes with fMRI data for Subject 1 at the moment.

Example usage: python train_kh.py --name test --rois LOC --subject_number 1 --resolution 256 --feature_extractor alexnetconv5 --l2 0

## Prediction
To get predictions from a trained encoder on new stimuli, run the `predict.py` script (see script for arguments).
This will save the results for each stimulus as an `.npy` file, which you can then load using `np.load(*.npy)`
for further analysis.

Example usage: python predict.py --name LOC_fc6 --stimuli_folder images --save_folder LOC_fc6 
