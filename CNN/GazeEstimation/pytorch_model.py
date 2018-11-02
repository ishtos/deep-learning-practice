import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable


class GazeNet(nn.Module):

    def __init__(self):
        super(GazeNet, self).__init__()
        model = models.alexnet(pretrained=True)
        self.alexnet = model.features
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(256*13*13, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.alexnet(x)

        y = self.layer1(x)
        print(y.size())
        y = self.layer2(y)
        print(y.size())
        y = self.layer3(y)
        print(y.size())

        x = F.dropout(F.relu(F.mul(x, y)), 0.5)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)

        return x


# class Flatten(nn.Module):
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return x

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.01)

# class Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))