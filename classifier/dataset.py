from PIL import Image

import numpy as np

import torch
import torch.utils.data as tudata
import torchvision.transforms as transforms

import trf

import os
import pickle


class Dataset(tudata.Dataset):

    def __init__(self, root=None, tmp_file=None, phase=None, img_path=None):
        self.phase = phase
        self.root = root
        self.tmp_file = tmp_file

        if phase != 'classify':
            # get file name list
            self.files = []
            with open(tmp_file) as f:
                lines = f.readlines()
                for line in lines:
                    self.files.append(root + line[:-1])

            # get labels
            self.labels = []
            for i in range(len(self.files)):
                if self.files[i][22] == '/':
                    self.labels.append(int(self.files[i][21])-1)
                elif self.files[i][23] == '/':
                    self.labels.append(int(self.files[i][21:23])-1)
                elif self.files[i][24] == '/':
                    self.labels.append(int(self.files[i][21:24])-1)
            self.labels = torch.LongTensor(self.labels)

        self.img_path = img_path

    def __getitem__(self, idx):
        if self.phase == 'classify':
            img = Image.open(self.img_path)
            img = img.convert('RGB')
            img = trf.resize(img, (64, 64))
            img = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                     std=(0.2675, 0.2565, 0.2761))
                ])(img)
            return img, 1
        img, label = self.files[idx], self.labels[idx]
        img = Image.open(img)
        img = img.convert('RGB') 
        if self.phase == 'train':
            img = trf.resize(img, (64, 64))
            img = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                     std=(0.2675, 0.2565, 0.2761))
                ])(img)
        if self.phase == 'test':
            img = trf.resize(img, (64, 64))
            img = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                     std=(0.2675, 0.2565, 0.2761))
                ])(img)
        
        return img, label

    def __len__(self):
        if self.phase == 'classify':
            return 1
        return len(self.labels)

#dataset_trn = Dataset(phase=train)
#dataset_val = Dataset(phase=val)
#dataloader_trn = tudata.DataLoader(
#    dataset_trn, batch_size=64, shuffle=True)
#dataloader_val = tudata.DataLoader(
#    dataset_val, batch_size=64, shuffle=False)
