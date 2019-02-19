# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import idenUnit, poolUnit


class ShuffleNet(nn.Module):
    def __init__(self, output_size, scale_factor = 1, g = 8):
        super(ShuffleNet, self).__init__()
        self.g = g
        # self.cs = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
        self.cs = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

        # compute output channels for stages
        c2 = self.cs[self.g]
        c2 = int(scale_factor * c2)
        c3, c4 = 2*c2, 4*c2

        # first conv layer & last fc layer
        self.conv1 = nn.Conv2d(3, 24, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(24)

        self.fc = nn.Linear(c4, output_size)

        # build stages
        self.stage2 = self.build_stage(24, c2, repeat_time = 3, first_group = False, downsample = False)
        self.stage3 = self.build_stage(c2, c3, repeat_time = 7)
        self.stage4 = self.build_stage(c3, c4, repeat_time = 3)

        # weights init
        self.weights_init()


    def build_stage(self, input_channel, output_channel, repeat_time, first_group = True, downsample = True):
        stage = [poolUnit(input_channel, output_channel, self.g, first_group = first_group, downsample = downsample)]
        
        for i in range(repeat_time):
            stage.append(idenUnit(output_channel, self.g))

        return nn.Sequential(*stage) 



    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        x = F.relu(self.bn1(self.conv1(inputs)))
        # x = F.max_pool2d(x, kernel_size = 3, stride = 2, padding = 1)
        # assert x.shape[1:] == torch.Size([24,56,56])

        # bottlenecks
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # print(x.shape)

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def test():
    from count import measure_model
    import numpy as np

    x = np.random.randn(10, 3, 32, 32).astype(np.float32)
    x = torch.from_numpy(x)

    net = ShuffleNet(10, g = 1, scale_factor = 0.5)
    f, c = measure_model(net, 32, 32)
    print("model size %.4f M, ops %.4f M" %(c/1e6, f/1e6))