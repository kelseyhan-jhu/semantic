from torch import nn
from torchvision.models.alexnet import alexnet


class AlexNetFC6(nn.Module):

    def __init__(self):
        super().__init__()

        base = alexnet(pretrained=True)

        self.conv = base.features
        self.avgpool = base.avgpool
        self.fc_6 = base.classifier[:3]

        self.eval()

    def forward(self, stimuli):
        x = self.conv(stimuli)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_6(x)
        return x
