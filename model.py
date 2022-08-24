import torch
import torch.nn.functional as F
from torch import nn



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,64, 4,3, 1)
        self.conv2 = nn.Conv2d(64, 128, 4,2,1)
        self.conv3 = nn.Conv2d(128, 256,3,2,1)
        self.maxpool = nn.MaxPool2d(2,2)


        self.fc1 = nn.Linear(256*1*1, 32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x= torch.relu(self.fc1(x))
        x =  torch.relu(self.fc2(x))
        output = torch.relu(self.fc3(x))
        return output
# model = CNN()
# fake_data = torch.rand(64, 1,28,28)
# img = fake_data[0].view(64,256)
# # print(img.shape)
# pre = model.forward(img)
# print(pre)
