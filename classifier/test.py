import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tudata
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from dataset import Dataset
from vgg import VGG19
import trf

import os
import argparse

BATCH_SIZE = 16
NUM_WORKERS = 8

parser = argparse.ArgumentParser(description='training vgg19')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint', default='./ckpt.pth', help='start from chp')
args = parser.parse_args()

# Model
print("==> Building Model..")
net = VGG19(num_class=172)

if args.resume:
    print("Resuming from checkpoint..")
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    best_loss = float('inf')
    start_epoch = 0

# Dataset
print("==> Loading Dataset..")

valset = Dataset(root='./ready_chinese_food',
                 tmp_file='./SplitAndIngreLabel/TE.txt', phase='test')

valloader = tudata.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

if torch.cuda.is_available():
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# val
def val(epoch):
    val_loss = 0

    for batch_idx, (inputs, labels) in enumerate(valloader):
        if (batch_idx > 20):
            break
        inputs = Variable(inputs.cuda()) if torch.cuda.is_available() else Variable(inputs)
        labels = Variable(labels.cuda()) if torch.cuda.is_available() else Variable(labels)

        outputs = net(inputs)
        pred = outputs.max(1)[1]
        pred = np.array(pred)
        labels = np.array(labels)
        acc = (pred == labels) / len(labels)
        print(acc)

        


for epoch in range(start_epoch, start_epoch+200):
    val(epoch)
