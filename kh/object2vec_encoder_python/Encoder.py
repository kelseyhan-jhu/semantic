import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, feature_extractor, encoder_weights):
        super().__init__()

        self.feature_extractor = feature_extractor

        if isinstance(encoder_weights, np.ndarray):
            encoder_weights = torch.from_numpy(encoder_weights)
        self.encoder = nn.Linear(in_features=encoder_weights.size(0),
                                 out_features=encoder_weights.size(1),
                                 bias=False)
        self.encoder.weight.data = encoder_weights

        self.eval()

    def forward(self, stimuli):
        features = self.feature_extractor(stimuli)
        voxels = self.encoder(features)
        return voxels
