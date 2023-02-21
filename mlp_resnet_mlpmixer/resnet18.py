'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot
import numpy as np
import os


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super().__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.nn.functional.relu(out + self.shortcut(x))
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        out = self.conv1(images)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # https://discuss.pytorch.org/t/adaptive-avg-pool2d-vs-avg-pool2d/27011
        # avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # out = avgpool(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def visualize(self, logdir):
        """ Visualize the kernel in the desired directory """
        # retrieve weights from the first hidden layer
        # print(self.conv1.get_parameter('weight').shape)
        filters = self.conv1.get_parameter('weight')
        # print(self.conv1.get_parameter('weight').data.shape) = [64, 3, 3, 3] =[blocks, patch, patch, channel]
        # Detach the tensor to avoid breaking the computation graph for numpy
        filters1 = filters.detach()
        # Numpy don't work on cuda
        # Numpy conversion is needed to plot the graph
        filters1 = filters1.cpu()
        filters1 = filters1.data.numpy()
        # print(filters1.shape)
        # # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters1.min(), filters1.max()
        filters1 = (filters1 - f_min) / (f_max - f_min)
        filter_mean = np.mean(filters1, axis=3)
        # print(filter_mean.shape)
        # plot first few filters
        n_filters, ix = 1, 8
        for i in range(ix):
            # get the filter
            for j in range(8):
                f = filter_mean[n_filters-1, :, :]
                ax = pyplot.subplot(ix, 8, n_filters)
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.imshow(f, cmap='gray')
                n_filters += 1
        # show the figure
        fig1 = pyplot.gcf()
        pyplot.show()
        pyplot.draw()
        fig1.savefig(os.path.join(logdir, f'visualize_kernel.png'))
