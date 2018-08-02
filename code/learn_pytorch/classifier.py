#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : classifier.py
# @Author: harry
# @Date  : 18-8-1 下午7:04
# @Desc  : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os

MODEL_FILE = './model/cifar_cnn.pkl'


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyModel():
    def __init__(self):
        # CNN
        self.net = Net()

        # Try to train on GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)
        self.net.to(self.device)

        # Dataset
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4,
                                                       shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                      shuffle=False, num_workers=2)
        self.classes = classes = ('plane', 'car', 'bird', 'cat',
                                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        # iterate over training data
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')

        # save model
        torch.save(self.net.state_dict(), MODEL_FILE)

    def load_model(self):
        if os.path.isfile(MODEL_FILE):
            self.net.load_state_dict(torch.load(MODEL_FILE))
            return True
        else:
            return False

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def test_summary(self):
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    model = MyModel()
    if model.load_model():
        print("Model weights loaded from {}".format(MODEL_FILE))
    else:
        print("No model file detected, training from scratch")
        model.train()
    model.test()
    model.test_summary()
