import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

#resnet18
def resnet18(num_classes=1,pretrained=False,num_channels=1):
    model = models.resnet18(pretrained=pretrained)
    model.fc=nn.Linear(model.fc.in_features,num_classes,bias=True)
    init.kaiming_normal_(model.fc.weight,nonlinearity='relu')
    init.constant_(model.fc.bias,0.01)
    # model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    return model

def resnet34(num_classes=1,pretrained=False,num_channels=1):
    model = models.resnet34(pretrained=pretrained)
    model.fc=nn.Linear(model.fc.in_features,num_classes,bias=True)
    init.kaiming_normal_(model.fc.weight,nonlinearity='relu')
    init.constant_(model.fc.bias,0.01)
    # model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    return model

def resnet50(num_classes=1,pretrained=False,num_channels=1):
    model = models.resnet50(pretrained=pretrained)
    model.fc=nn.Linear(model.fc.in_features,num_classes,bias=True)
    init.kaiming_normal_(model.fc.weight,nonlinearity='relu')
    init.constant_(model.fc.bias,0.01)
    # model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    return model

#vgg
def vgg11(num_classes=1,pretrained=False,num_channels=1):
    model = models.vgg11_bn(pretrained=pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,num_classes,bias=True)
    init.kaiming_normal_(model.classifier[-1].weight,nonlinearity='relu')
    init.constant_(model.classifier[-1].bias,0.01)
    # model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # init.kaiming_normal_(model.features[0].weight, mode='fan_out', nonlinearity='relu')
    return model