import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ScrutariOcularisModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(ScrutariOcularisModel, self).__init__()
        self.resnet =  models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Modificar la última capa para tener num_classes clases de salida
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)