import torch

import torch.nn as nn
import torch.nn.functional as F

from lib.cnn.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC
from lib.cnn.models.inception import InceptionD, InceptionE


class InceptionEnv(nn.Module):

    def __init__(self, n_labels=4520, n_input=77, dropout=0.7, last_layer=True, logit=False, exp=False,
                 normalize_weight=1.):
        super(InceptionEnv, self).__init__()
        if n_input >= 15:
            self.Conv2d_1a_3x3 = BasicConv2d(n_input, 80, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2a_3x3 = BasicConv2d(80, 80, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2b_3x3 = BasicConv2d(80, 100, kernel_size=3, padding=1, stride=1)
            self.Conv2d_3b_1x1 = BasicConv2d(100, 124, kernel_size=1)
            self.Conv2d_4a_3x3 = BasicConv2d(124, 192, kernel_size=3, padding=1, stride=1)
        else:
            self.Conv2d_1a_3x3 = BasicConv2d(n_input, 32, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1, stride=1)
            self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
            self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1, stride=1)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, n_labels)

        self.dropout = dropout
        self.last_layer = last_layer
        self.logit = logit
        self.exp = exp

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if poisson loss, for instance...
        if normalize_weight != 1.:
            for p in self.parameters():
                p.data.div_(normalize_weight)

    def forward(self, x):
        # (80, 3) x 64 x 64
        x = self.Conv2d_1a_3x3(x)

        # (124, 32) x 64 x 64
        x = self.Conv2d_2a_3x3(x)

        # (124, 32) x 64 x 64
        x = self.Conv2d_2b_3x3(x)

        # (124, 32) x 64 x 64
        x = self.Conv2d_3b_1x1(x)

        # (124, 80) x 64 x 64
        x = self.Conv2d_4a_3x3(x)

        # 192 x 64 x 64
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # 192 x 32 x 32
        x = self.Mixed_5b(x)

        # 256 x 32 x 32
        x = self.Mixed_5c(x)

        # 288 x 32 x 32
        x = self.Mixed_5d(x)

        # 288 x 32 x 32
        x = self.Mixed_6a(x)

        # 768 x 15 x 15
        x = self.Mixed_6b(x)

        # 768 x 15 x 15
        x = self.Mixed_6c(x)

        # 768 x 15 x 15
        x = self.Mixed_6d(x)

        # 768 x 15 x 15
        x = self.Mixed_6e(x)

        # 768 x 15 x 15
        x = self.Mixed_7a(x)

        # 1280 x 7 x 7
        x = self.Mixed_7b(x)

        # 2048 x 7 x 7
        x = self.Mixed_7c(x)

        # 2048 x 7 x 7
        x = F.avg_pool2d(x, kernel_size=7)

        # 1 x 1 x 2048
        x = F.dropout(x, p=self.dropout, training=self.training)  # increased dropout probability

        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)

        if self.last_layer:
            # 2048
            x = self.fc(x)
            # (num_classes)
            if not self.training and not self.logit and not self.exp:
                x = F.softmax(x, dim=-1)
            elif not self.training and not self.logit:
                x = x.exp()

        return x
    
    def __repr__(self):
        return '(Environmental Inception)'
