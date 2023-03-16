import random
# Pytorch
import torch
import torch.nn as nn
# Local
from modules import compress_jpeg, decompress_jpeg


class DiffJPEG(nn.Module):
    def __init__(self, height: int,
                 width: int,
                 differentiable: bool = True,
                 quality: int = 80,
                 p: float = 1.):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
            q(float): Possibility to conduct compression.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round

        assert 0 < quality < 100, f'Error, quality must be in range(0, 100), but got {quality}'
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)
        self.p = p

    def forward(self, x):
        if random.random() <= self.p:
            y, cb, cr = self.compress(x)
            recovered = self.decompress(y, cb, cr)
        else:
            recovered = x
        return recovered


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.


def run():
    jpeg = DiffJPEG(128, 128, differentiable=False)
    pass


if __name__ == '__main__':
    run()
