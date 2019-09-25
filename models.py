import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, wavelet, levels=0, grayscale=False):
        super(SRCNN, self).__init__()
        self.levels = levels
        self.wavefilters = nn.Parameter(construct_wt_filters(wavelet),
                                        requires_grad=False)
        subbands = 4**levels
        in_ch = 3 ** (1 - int(grayscale)) * subbands
        self.fc_res = int((32 / 2 ** 2) / 2 ** levels)

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
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        if self.levels > 0:
            x = wt(x, self.wavefilters, self.levels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * self.fc_res * self.fc_res)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


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


def construct_wt_filters(wavelet):
    assert wavelet in pywt.wavelist(), 'Invalid wavelet choice'

    # A familia de wavelet
    w=pywt.Wavelet(wavelet)

    # do fliplr e transforma em tensor
    dec_hi = torch.Tensor(w.dec_hi[::-1])
    dec_lo = torch.Tensor(w.dec_lo[::-1])

    # calcula produtos externos entre todos formando: LL, LH, HL e HH; e empilha tudo em um tensor
    filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    return filters


def wt(x, filters, levels=1):
    b, c, h, w = x.shape

    # padding ora em cima e a esquerda, ora embaixo e a direita
    p1, p2 = (levels + 1) % 2, levels % 2

    # DWT deve ser aplicada separadamente em cada canal RGB
    x = x.view(b*c, 1, h, w)
    xpad = F.pad(x, (p1, p2, p1, p2), mode='reflect')

    # já faz o downsampling de 2 na altura e largura
    x = F.conv2d(xpad, filters[:, None], stride=2)
    if levels > 1:
        # se for decomposição em vários níveis chama a func recursivamente
        x = wt(x, filters, levels-1)

    # organiza tudo novamente com os mesmo nb inicial de imgs no batch e os coeffs na dim dos canais
    x = x.view(b,-1, *x.shape[2:])
    return x

# Original:
# init biases to 1
# init weights from N(0, 0.01^2)

# Here (default):
# init bias as Lecunn: U(-k,k) w/k = sqrt(1/fan_in)
# init weights as kaiming_uniform: U(-k,k) w/k = sqrt(6/fan_in)