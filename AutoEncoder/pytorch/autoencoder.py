import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.resnet34 = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3,
            resnet34.layer4,
        )

    def forward(self, x):
        x = self.resnet34(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
        )
        self.layer5 = nn.Sequential(
           nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.layer1(x))
        x = self.upsample(x)
        x = F.relu(self.layer2(x))
        x = self.upsample(x)
        x = F.relu(self.layer3(x))
        x = self.upsample(x)
        x = F.relu(self.layer4(x))
        x = self.upsample(x)
        x = F.sigmoid(self.layer5(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x