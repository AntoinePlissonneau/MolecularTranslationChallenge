# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import (Compose, OneOf, Normalize, Resize)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from configparser import ConfigParser

import sys
sys.path.append('./AutoEncoder')
from Trainer_AE import *

# ====================================================
# preprocessing.ini
# ====================================================
config = ConfigParser()
config.read('./preprocessing.ini')
size = int(config['Preprocessing_parameters']['size_image'])


config = ConfigParser()
config.read('./params_AE.ini')

# general parameters
params_ae = {'data_name': config['Data_parameters']['data_name'],
          'n_channels': int(config['Model_parameters']['n_channels']),
          'output_dim': int(config['Model_parameters']['output_dim']),
          'start_epoch': int(config['Training_parameters']['start_epoch']),
          'epochs': int(config['Training_parameters']['epochs']),
          'epochs_since_improvement': int(config['Training_parameters']['epochs_since_improvement']),
          'model_lr': float(config['Training_parameters']['model_lr']),
          'grad_clip': eval(config['Training_parameters']['grad_clip']),
          'best_mse': float(config['Training_parameters']['best_mse']),
          'print_freq': int(config['Training_parameters']['print_freq']),
          'checkpoint': (config['Training_parameters']['checkpoint']=='True'),
          'checkpoint_path':config['Training_parameters']['checkpoint_path']
         }

mol = Trainer_AE(params_ae)

# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.transform_ae = transform[0]
        self.transform_lstm = transform[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = ((255-cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))/255).astype(np.float32)

        #Transformation before the autoencoder
        augmented = self.transform_ae(image=image)
        image = augmented['image']
        image = mol.predict(image, numpy=True)

        #Transformation before Resnet
        augmented_ = self.transform_lstm(image=image)
        image = augmented_['image']

        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def bms_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)

def get_transforms(*, data):

    if data == 'train':
        return (Compose([Resize(size, size), ToTensorV2(),]),
                Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],), ToTensorV2(),]))

    elif data == 'valid':
        return (Compose([Resize(size, size), ToTensorV2(),]),
                Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],), ToTensorV2(),]))