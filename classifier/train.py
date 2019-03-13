import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tudata
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms

from dataset import Dataset
from vgg import VGG19
import trf

import argparse

BATCH_SIZE = 32
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

trainset = Dataset(root='./ready_chinese_food',
                   tmp_file='./SplitAndIngreLabel/TR.txt', phase='train')
valset = Dataset(root='./ready_chinese_food',
                 tmp_file='./SplitAndIngreLabel/TE.txt', phase='test')

trainloader = tudata.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valloader = tudata.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

if torch.cuda.is_available():
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


# train
def train(epoch):
    print("Epoch: %d" % epoch)
    net.train()
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if (batch_idx > 500):
            break
        inputs = Variable(inputs.cuda()) if torch.cuda.is_available() else Variable(inputs)
        labels = Variable(labels.cuda()) if torch.cuda.is_available() else Variable(labels)

        optimizer.zero_grad()
        print('prepared to calculate')
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data.item(), train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))


# val
def val(epoch):
    net.eval()
    val_loss = 0

    # Save checkpoint
    global best_loss
    val_loss /= len(valloader)
    if val_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = val_loss

    for batch_idx, (inputs, labels) in enumerate(valloader):
        if (batch_idx > 20):
            break
        inputs = Variable(inputs.cuda()) if torch.cuda.is_available() else Variable(inputs)
        labels = Variable(labels.cuda()) if torch.cuda.is_available() else Variable(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.data.item()
        print('val_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data.item(), val_loss/(batch_idx+1), batch_idx+1, len(valloader)))


for epoch in range(start_epoch, start_epoch+200):
    print('==> Start training..')
    train(epoch)
    print('==> Start validating..')
    test(epoch)
