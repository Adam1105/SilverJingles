from functools import partial

from torch import nn
from torch.nn import Module, ModuleList
from torch.functional import F
from torch.nn.utils import weight_norm


class ResolutionDiscriminator(Module):
    def __init__(self, n_fft, hop_length, win_length,
                 convs=nn.ModuleList([weight_norm(nn.Conv2d(1, int(32 * 1), (3, 9), padding=(1, 4))),
                                      weight_norm(nn.Conv2d(int(32 * 1), int(32), (3, 9), stride=(1, 2), padding=(1, 4))),
                                      weight_norm(nn.Conv2d(int(32), int(32), (3, 9), stride=(1, 2), padding=(1, 4))),
                                      weight_norm(nn.Conv2d(int(32), int(32), (3, 9), stride=(1, 2), padding=(1, 4))),
                                      weight_norm(nn.Conv2d(int(32), int(32), (3, 3), padding=(1, 1))),]),
                 conv_post=weight_norm(nn.Conv2d(int(32), 1, (3, 3), padding=(1, 1))),
                 activation=partial(F.leaky_relu, negative_slope=0.1)):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.convs = convs
        self.conv_post = conv_post
        self.activation = activation

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(Module):
    def __init__(self, discriminators: ModuleList[ResolutionDiscriminator]):
        super().__init__()
        self.discriminators = discriminators

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
