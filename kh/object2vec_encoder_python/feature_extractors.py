from torch import nn
from torchvision.models.alexnet import alexnet


class AlexNetFC6(nn.Module):

    def __init__(self):
        super().__init__()

        base = alexnet(pretrained=True)
        print(base)
        self.conv = base.features
        self.avgpool = base.avgpool
        self.fc_6 = base.classifier[:3]

        self.eval()

    def forward(self, stimuli):
        x = self.conv(stimuli)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_6(x) # shape: [10, 4096]
        return x

class AlexNetConv5(nn.Module):
    def __init__(self):
        super().__init__()

        base = alexnet(pretrained=True)
        self.conv = base.features
        self.conv1 = base.features[:2]
        self.avgpool = base.avgpool
        
        self.eval()

    def forward(self, stimuli):
        x = self.conv(stimuli) #shape: [stim, 256, 7, 7] / mean test set r = 0.7297
        #x = self.avgpool(x) #shape: [stim, 256, 6, 6]
        x = x.view(x.size(0), -1) #shape: [stim, 12544] / mean test set r =  0.7218
        #x = self.fc_6(x) #shape: [stim, 4096] / mean test set r = 0.7142
        return x