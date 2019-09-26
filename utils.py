import torch
import torch.nn.functional as F
import pywt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def construct_wt_filters(wavelet):
    assert wavelet in pywt.wavelist(), 'Invalid wavelet choice {}'.format(wavelet)

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