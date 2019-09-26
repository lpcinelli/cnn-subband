import torch.nn as nn
import torch.nn.functional as F
from utils import construct_wt_filters, wt


class SRCNN(nn.Module):
    def __init__(self, nb_classes, wavelet, levels=0, grayscale=False):
        super(SRCNN, self).__init__()
        self.levels = levels
        self.wavefilters = nn.Parameter(construct_wt_filters(wavelet),
                                        requires_grad=False)
        subbands = 4**levels
        in_ch = 3**(1 - int(grayscale)) * subbands
        self.fc_res = int((32 / 2**2) / 2**levels)
        self.leakage = 0.1

        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1, groups=subbands)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, groups=subbands)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, groups=subbands)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, groups=subbands)
        self.conv5 = nn.Conv2d(512, 128, 3, padding=1, groups=subbands)
        # self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * self.fc_res * self.fc_res, 4096)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, nb_classes)

    def forward(self, x):
        if self.levels > 0:
            x = wt(x, self.wavefilters, self.levels)
        x = F.leaky_relu(self.conv1(x), negative_slope=self.leakage)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.leakage)
        x = self.pool(F.leaky_relu(self.conv3(x), negative_slope=self.leakage))

        x = F.leaky_relu(self.conv4(x), negative_slope=self.leakage)
        x = self.pool(F.leaky_relu(self.conv5(x), negative_slope=self.leakage))

        x = x.view(-1, 128 * self.fc_res * self.fc_res)
        x = self.drop(F.leaky_relu(self.fc1(x), negative_slope=self.leakage))
        x = self.drop(F.leaky_relu(self.fc2(x), negative_slope=self.leakage))
        x = self.fc3(x)
        return x

    @classmethod
    def from_name(cls, num_classes, wavelet, levels=0, grayscale=False):
        wavelet=kwargs.get('wavelet', None)
        levels=kwargs.get('level', 0)
        grayscale=kwargs.get('grayscale', None)

        return cls(num_classes, wavelet, levels, grayscale)

# Original:
# init biases to 1
# init weights from N(0, 0.01^2)

# Here (default):
# init bias as Lecunn: U(-k,k) w/k = sqrt(1/fan_in)
# init weights as kaiming_uniform: U(-k,k) w/k = sqrt(6/fan_in)
