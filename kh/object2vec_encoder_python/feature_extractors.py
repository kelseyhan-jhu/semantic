from torch import nn
#from torchvision.models.alexnet import alexnet
import torchvision.models as models


class AlexNetFull(nn.Module):
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
#         print(conv1.shape)
#         print(conv2.shape)
#         print(conv3.shape)
#         print(conv4.shape)
#         print(conv5.shape)
#         print(fc6.shape)
#         print(fc7.shape)
        return conv1, conv2, conv3, conv4, conv5, fc6, fc7
    
    
# class AlexNetConv5(nn.Module):
#     def __init__(self):
#         super().__init__()
#         base = models.alexnet(pretrained=False)
#         self.conv = base.features
#         self.conv5 = base.features[:12]
#         self.avgpool = base.avgpool
        
#         self.eval()

#     def forward(self, stimuli):
#         x = self.conv(stimuli) #shape: [stim, 256, 7, 7]
#         #x = self.avgpool(x) #shape: [stim, 256, 6, 6]
#         x = x.view(x.size(0), -1) #shape: [stim, 12544]
#         #x = self.fc_6(x) #shape: [stim, 4096]
#         return x
    
class AlexNetConv1(nn.Module):
    def __init__(self):
        super().__init__()
        base = alexnet(pretrained=True)
        #self.conv = base.features
        self.conv1 = base.features[:2]
        #self.avgpool = base.avgpool
        
        self.eval()

    def forward(self, stimuli):
        x = self.conv1(stimuli) #shape: [stim, 256, 7, 7]
        #x = self.avgpool(x) #shape: [stim, 256, 6, 6]
        x = x.view(x.size(0), -1) #shape: [stim, 12544]
        #x = self.fc_6(x) #shape: [stim, 4096]
        return x

class AlexNetConv5(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.alexnet(pretrained=True)
        self.conv = base.features
        #self.avgpool = base.avgpool
        
        self.eval()

    def forward(self, stimuli):
        x = self.conv(stimuli) #shape: [stim, 256, 7, 7]
        #x = self.avgpool(x) #shape: [stim, 256, 6, 6]
        x = x.view(x.size(0), -1) #shape: [stim, 12544]
        #x = self.fc_6(x) #shape: [stim, 4096]
        return x
    
class AlexNetFC6(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.alexnet(pretrained=True)

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
    

class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.vgg16(pretrained=pretrained)
        #self.conv1 = base.features[0:5]
        #self.conv2 = base.features[5:10]
        #self.conv3 = base.features[10:17]
        #self.conv4 = base.features[17:24]
        #self.conv5 = base.features[24:]
        #self.avgpool = base.avgpool
        #self.fc6 = base.classifier[:3]
        #self.fc7 = base.classifier[3:5]
        self.conv = base.features
        self.eval()

    def forward(self, stimuli):
        #conv1 = self.conv1(stimuli)
        #conv2 = self.conv2(conv1)
        #conv3 = self.conv3(conv2)
        #conv4 = self.conv4(conv3)
        #conv5 = self.conv5(conv4)
        #avgpool = self.avgpool(conv5)
        #avgpool = avgpool.view(avgpool.size(0), -1)
        #fc6 = self.fc6(avgpool)
        #fc7 = self.fc7(fc6)
        x = self.conv(stimuli)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        return x
    
class VGG16_conv5(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.vgg16(pretrained=pretrained)
        #self.conv1 = base.features[0:5]
        #self.conv2 = base.features[5:10]
        #self.conv3 = base.features[10:17]
        #self.conv4 = base.features[17:24]
        #self.conv5 = base.features[24:]
        #self.avgpool = base.avgpool
        #self.fc6 = base.classifier[:3]
        #self.fc7 = base.classifier[3:5]
        self.conv = base.features
        self.eval()

    def forward(self, stimuli):
        #conv1 = self.conv1(stimuli)
        #conv2 = self.conv2(conv1)
        #conv3 = self.conv3(conv2)
        #conv4 = self.conv4(conv3)
        #conv5 = self.conv5(conv4)
        #avgpool = self.avgpool(conv5)
        #avgpool = avgpool.view(avgpool.size(0), -1)
        #fc6 = self.fc6(avgpool)
        #fc7 = self.fc7(fc6)
        x = self.conv(stimuli)
        #print(x.shape)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        return x
    
class VGG16_post_conv5(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.vgg16(pretrained=True)
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.eval()

    def forward(self, conv5):
        avgpool = self.avgpool(conv5)
        avgpool = avgpool.view(avgpool.size(0), -1)
        x = self.classifier(avgpool)
        #print(x.shape)
        return x