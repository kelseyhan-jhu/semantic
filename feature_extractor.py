import torch
import torch.nn as nn
from torchvision import models
from utils import image_to_tensor, listdir
from tqdm import tqdm
import os
from argparse import ArgumentParser

class AlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.alexnet(pretrained=pretrained)
        self.conv1 = base.features[:3]
        self.conv2 = base.features[3:6]
        self.conv3 = base.features[6:8]
        self.conv4 = base.features[8:10]
        self.conv5 = base.features[10:]
        self.avgpool = base.avgpool
        self.fc6 = base.classifier[:3]
        self.fc7 = base.classifier[3:5]
        self.eval()

    def forward(self, stimuli):
        conv1 = self.conv1(stimuli)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        avgpool = self.avgpool(conv5)
        avgpool = avgpool.view(avgpool.size(0), -1)
        fc6 = self.fc6(avgpool) # shape: [10, 4096]
        fc7 = self.fc7(fc6)
        return fc7

class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.vgg16(pretrained=pretrained)
        self.conv1 = base.features[0:5]
        self.conv2 = base.features[5:10]
        self.conv3 = base.features[10:17]
        self.conv4 = base.features[17:24]
        self.conv5 = base.features[24:]
        self.avgpool = base.avgpool
        self.fc6 = base.classifier[:3]
        self.fc7 = base.classifier[3:5]
        self.eval()

    def forward(self, stimuli):
        conv1 = self.conv1(stimuli)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        avgpool = self.avgpool(conv5)
        avgpool = avgpool.view(avgpool.size(0), -1)
        fc6 = self.fc6(avgpool)
        fc7 = self.fc7(fc6)
        return fc7

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


if __name__ == '__main__':
    parser = ArgumentParser(description='Feature extractor')
    parser.add_argument('--model', required=True, type=str,
                        help='Name of the feature extractor (Options include: alexnet, vgg)')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Name of the dataset to extract features from (Options include: object2vec, things, imagenet)')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution at which to resize stimuli')
    args = parser.parse_args()

    #get_features(args.model, args.dataset, args.resolution)

    #def get_features(model, dataset, resolution):
    if args.model == 'alexnet':
        feature_extractor = AlexNet()
    elif args.model == 'vgg':
        feature_extractor = VGG16()
    else:
        raise ValueError('Unimplemented feature extractor: {}'.format(args.model))
    
    feature_extractor.conv1.register_forward_hook(get_activation('conv1'))
    feature_extractor.conv2.register_forward_hook(get_activation('conv2'))
    feature_extractor.conv3.register_forward_hook(get_activation('conv3'))
    feature_extractor.conv4.register_forward_hook(get_activation('conv4'))
    feature_extractor.conv5.register_forward_hook(get_activation('conv5'))
    feature_extractor.fc6.register_forward_hook(get_activation('fc6'))
    feature_extractor.fc7.register_forward_hook(get_activation('fc7'))

    conditions = listdir('data/image/' + args.dataset)
    for c in tqdm(conditions):
        stimuli = listdir(c)
        c_name = c.split('/')[-1]
        os.mkdir('processed/feature/' + args.dataset + '/' + args.model + '/' + c_name)
        stimuli_tensor = [image_to_tensor(s, resolution=args.resolution) for s in stimuli]
        for name, tensor in zip(stimuli, stimuli_tensor):
            activation = {}
            output = feature_extractor(tensor.unsqueeze(0))
            file = name.split('/')[-1] + '.pth'
            torch.save(activation, os.path.join('processed/feature', args.dataset, args.model, c_name, file))
