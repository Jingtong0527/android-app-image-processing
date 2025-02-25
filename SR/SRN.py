import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(c1)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m

class ResidualBlock_ESA(nn.Module):
    '''
    ---Conv-ReLU-Conv-ESA +-
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_ESA, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.ESA = ESA(nf, nn.Conv2d)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = (self.conv1(x))
        out = self.lrelu(out)
        out = (self.conv2(out))
        out = self.ESA(out)
        return out
# class ResidualBlock_ESA_1(nn.Module):
#     '''
#     ---Conv-ReLU-Conv-ESA +-
#     '''
#     def __init__(self, nf=64):
#         super(ResidualBlock_ESA_1, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.ESA = ESA(nf, nn.Conv2d)
#         self.lrelu = nn.LeakyReLU(0.1)
#
#     def forward(self, x):
#         out = (self.conv1(x))
#         out = self.lrelu(out)
#         out = (self.conv2(out))
#         out = self.ESA(out)
#         return out
# class ResidualBlock_ESA_2(nn.Module):
#     '''
#     ---Conv-ReLU-Conv-ESA +-
#     '''
#     def __init__(self, nf=64):
#         super(ResidualBlock_ESA_2, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.ESA = ESA(nf, nn.Conv2d)
#         self.lrelu = nn.LeakyReLU(0.1)
#
#     def forward(self, x):
#         out = (self.conv1(x))
#         out = self.lrelu(out)
#         out = (self.conv2(out))
#         out = self.ESA(out)
#         return out
# class ResidualBlock_ESA_3(nn.Module):
#     '''
#     ---Conv-ReLU-Conv-ESA +-
#     '''
#     def __init__(self, nf=64):
#         super(ResidualBlock_ESA_3, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.ESA = ESA(nf, nn.Conv2d)
#         self.lrelu = nn.LeakyReLU(0.1)
#
#     def forward(self, x):
#         out = (self.conv1(x))
#         out = self.lrelu(out)
#         out = (self.conv2(out))
#         out = self.ESA(out)
#         return out
class SRN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, upscale=4):
        super(SRN, self).__init__()
        nf = 64
        blocks = 4
        basic_block = functools.partial(ResidualBlock_ESA, nf=nf)
        self.head = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.tail = nn.Conv2d(nf, out_nc * upscale * upscale, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.recon_trunk_0 = ResidualBlock_ESA()
        self.recon_trunk_1 = ResidualBlock_ESA()
        self.recon_trunk_2 = ResidualBlock_ESA()
        self.recon_trunk_3 = ResidualBlock_ESA()

    def forward(self, x):
        fea = self.head(x)
        out = fea
        # layer_names = self.recon_trunk._modules.keys()

        fea = self.recon_trunk_0(fea)
        fea = self.recon_trunk_1(fea)
        fea = self.recon_trunk_2(fea)
        fea = self.recon_trunk_3(fea)
        out = fea + out
        out = self.pixel_shuffle(self.tail(out))
        return out