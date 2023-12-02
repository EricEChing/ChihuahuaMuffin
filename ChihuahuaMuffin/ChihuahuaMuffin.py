import torch
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
import torch.nn as nn
import torch.optim as optim
from chiMufData import *

num_epochs = 20
BATCH_SIZE = 60

#REMINDER: muffin label=0, chihuahua label=1 (i think)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 256),  # Adjust based on your data dimensions
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x.view(1,-1))
        return x
    
