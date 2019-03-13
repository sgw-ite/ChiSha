from PIL import Image
import numpy as np
import torch
import torch.utils.data as tudata
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import pickle
import matplotlib.pyplot as plt


EPOCH = 10

# shape?
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class cifar10(tudata.Dataset):

    def __init__(self, train=True, transform=True):
        self.train = train
        self.transform = transform
        # get list
        data0, label0 = self._getdata(train=train)
        # get array
        data0 = np.array(data0)
        label0 = np.array(label0)
        # reshape data
        # height*width*channels for Image process
        data0 = data0.reshape(-1, 32, 32, 3)
        self.inputs = data0
        self.target = label0

    def __getitem__(self, index):
        img, label = self.inputs[index], self.target[index]
        img = Image.fromarray(img)
        if self.transform:
            img = transform(img)  # a tensor to network (int?)
        label = torch.from_numpy(np.array(label)).long()  # longtensor
        return img, label

    def __len__(self):
        return len(self.inputs)

    def _unpickle(self, file):
        '''
        type = dict
        '''
        with open(file, 'rb') as fo:
            _dict = pickle.load(fo, encoding='bytes')
        return _dict

    def _getdata(self, train=False):
        '''
        type = list
        '''
        if not train:
            fileoftest = './cifar10/test_batch'
            data_test = self._unpickle(fileoftest)
            test_data = data_test[b'data']
            test_labels = data_test[b'labels']
            return test_data, test_labels

        else:
            for i in range(1, 6):
                fileoftrain = './cifar10/data_batch_' + str(i)
                data_train = self._unpickle(fileoftrain)
                if i == 1:
                    train_data = data_train[b'data']
                    train_labels = data_train[b'labels']
                else:
                    train_data = np.concatenate(
                        (train_data, data_train[b'data']))
                    train_labels = np.concatenate(
                        (train_labels, data_train[b'labels']))
            return train_data, train_labels


my_train_dataset = cifar10(train=True)
my_test_dataset = cifar10(train=False)
mytrainloader = tudata.DataLoader(
    my_train_dataset, batch_size=64, shuffle=True)
mytestloader = tudata.DataLoader(my_test_dataset, batch_size=64, shuffle=True)


class Mynet(nn.Module):

    def __init__(self):
        super(Mynet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(6, 16, 5)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm1d(120)
        self.bn5 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(400, 120)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(120, 84)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(84, 10)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        # x = self.bn1(x)
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.bn2(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.bn3(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        # x = self.bn4(x)
        x = F.relu(self.fc2(x))
        # x = self.bn5(x)
        x = F.relu(self.fc3(x))
        return x


mynet = Mynet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mynet.parameters(), lr=0.03)

loss_his = []
for epoch in range(EPOCH):
    running_loss = 0.0

    for i, data in enumerate(mytrainloader):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = mynet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            loss_his.append(loss)

    acc_sum = 0.
    acc = 0.
    l_pred_sum = 0
    for i, data in enumerate(mytestloader):
        test_inputs, test_labels = data
        test_output = mynet(test_inputs)
        _, l_pred = torch.max(test_output, 1)
        acc = torch.sum(l_pred == test_labels)
        acc_sum += acc
        l_pred_sum += len(l_pred)
    acc_score = float(acc_sum) / l_pred_sum
    print('ACC:', acc_score)

# plt.plot(range(250),loss_his)
# plt.show()
torch.save(mynet.state_dict(), './state_dict')
