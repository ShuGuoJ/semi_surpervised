import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ) if in_channel!=out_channel else None


    def forward(self, input):
        o1 = torch.relu(self.bn1(self.conv1(input)))
        o2 = self.bn2(self.conv2(o1))
        o3 = o2+self.skip(input) if self.skip else o2+input
        return o3




class Resnet18(nn.Module):
    def __init__(self, in_channel, nc):
        super(Resnet18, self).__init__()
        channels = [64,128,256,512]
        self.conv1 = nn.Conv2d(in_channel, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2,1)
        in_channel = 64
        self.layer1 = self.make_layers(in_channel, channels[0], 2)
        self.layer2 = self.make_layers(channels[0], channels[1], 2)
        self.layer3 = self.make_layers(channels[1], channels[2], 2)
        self.layer4 = self.make_layers(channels[2], channels[3], 2)
        self.classifier = nn.Linear(channels[3], nc)

    def forward(self, input):
        o1 = torch.relu(self.bn1(self.conv1(input)))
        o1 = self.maxpool(o1)
        o1 = self.layer1(o1)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)
        o4 = F.adaptive_avg_pool2d(o4, (1,1))
        o4 = o4.view(o4.shape[0], -1)
        logits = self.classifier(o4)
        return logits

    def make_layers(self, in_channel, out_channel, depth):
        blocks = []
        for i in range(depth):
            if not i:
                blocks.append(BasicBlock(in_channel, out_channel, 2 if in_channel!=out_channel else 1))
            else:
                blocks.append(BasicBlock(out_channel, out_channel, 1))
        return nn.Sequential(*blocks)
