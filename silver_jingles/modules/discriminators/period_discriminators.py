from functools import partial

from torch import nn
from torch.nn import ModuleList, Module, Conv2d
from torch.functional import F
from torch.nn.utils import weight_norm


class PeriodDiscriminator(Module):
    def __init__(self, period,
                 convs=nn.ModuleList([
                        weight_norm(Conv2d(1, int(32), (5, 1), (3, 1), padding=(2, 0))),
                        weight_norm(Conv2d(int(32), int(128), (5, 1), (3, 1), padding=(2, 0))),
                        weight_norm(Conv2d(int(128), int(512), (5, 1), (3, 1), padding=(2, 0))),
                        weight_norm(Conv2d(int(512), int(1024), (5, 1), (3, 1), padding=(2, 0))),
                        weight_norm(Conv2d(int(1024), int(1024), (5, 1), 1, padding=(2, 0))),]),
                 conv_post=weight_norm(Conv2d(int(1024), 1, (3, 1), 1, padding=(1, 0))),
                 activation=partial(F.leaky_relu, negative_slope=0.1)):
        super().__init__()
        self.period = period
        self.convs = convs
        self.activation = activation
        self.conv_post = conv_post

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(Module):
    def __init__(self, discriminators: ModuleList[PeriodDiscriminator]):
        super().__init__()
        self.discriminators = discriminators

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
