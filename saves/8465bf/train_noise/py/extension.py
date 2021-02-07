import argparse
import os
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from misc_functions import *
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time


# define augmentation
def apply_transform(mode=None):
# same preprocessing as the original trained model for pneumonia
    if mode == 'train_noise':
        transform = T.Compose([T.Resize((256,256)),
                            T.RandomHorizontalFlip(),
                            T.RandomRotation((-20,+20)),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                            ])

    elif mode == 'test_noise' or mode == 'val_noise':
        transform = T.Compose([T.Resize((256,256)),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                            ])
        
    return transform

def get_arch():
    model = torchvision.models.vgg19(pretrained=True)
    # add Linear classifier layer
    in_features = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 2),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    return model
