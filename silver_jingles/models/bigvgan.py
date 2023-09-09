import torch.nn
from torch import nn

from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from silver_jingles.layers.conv import WeightNormConv1D, WeightNormConvTransposed1D
from silver_jingles.modules.discriminators.period_discriminators import PeriodDiscriminator
from silver_jingles.modules.discriminators.resolution_discriminators import ResolutionDiscriminator, \
    MultiResolutionDiscriminator
from silver_jingles.modules.fundamental_waveform_generators import InputUpSamplingResidualWaveformGenerator
from silver_jingles.modules.residual_blocks import ResBlock1
from silver_jingles.activations.alias_free import AliasFreeActivation1d
from silver_jingles.activations.snake import SnakeBeta, Snake


class BigVGAN(nn.Module):
    def __init__(self,
                 input_channel_size=100,
                 upsample_rates=(4, 4, 2, 2, 2, 2),
                 upsample_kernel_sizes=(8, 8, 4, 4, 4, 4),
                 upsample_initial_channel=1536,
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 activation_name="snakebeta",
                 snake_logscale=True):
        super().__init__()

        if activation_name == 'snake':
            activation_class = Snake
        elif activation_name == 'snakebeta':
            activation_class = SnakeBeta
        else:
            raise NotImplementedError(
                "Activation incorrectly specified. For now only 'snake', and 'snakebeta' are available.")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes[0]):
            raise RuntimeError("Number of resblock kernel sizes needs to match number of resblock dilation sizes")
        if any(len(resblock_dilation_sizes[0]) == len(d) for d in resblock_dilation_sizes):
            raise RuntimeError("Inconsistent dilation sizes. Each block of dilations should have the same length")
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise RuntimeError("Number of upsample rates needs to match number of upsample kernel sizes")

        self.num_layers = len(self.convs1) + len(self.convs2)
        pre_convolution = WeightNormConv1D(in_channels=input_channel_size, out_channels=upsample_initial_channel,
                                           kernel_size=7)
        # transposed conv-based upsamplers. does not apply anti-aliasing
        ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ups.append(nn.ModuleList([WeightNormConvTransposed1D(in_channels=upsample_initial_channel // (2 ** i),
                                                                 out_channels=upsample_initial_channel // (2 ** (i + 1)),
                                                                 kernel_size=k, stride=u)]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                convs1 = nn.ModuleList([WeightNormConv1D(in_channels=ch, out_channels=ch, kernel_size=k, dilation=d_i)
                                        for d_i in d])
                convs2 = nn.ModuleList([WeightNormConv1D(in_channels=ch, out_channels=ch, kernel_size=k) for _ in d])
                activations = nn.ModuleList(
                        [AliasFreeActivation1d(activation=activation_class(ch, alpha_logscale=snake_logscale))
                         for _ in range(self.num_layers)])
                resblocks.append(ResBlock1(convs1=convs1, convs2=convs2,
                                           activations1=activations[self.activations[::2]],
                                           activations2=activations[self.activations[1::2]]))
        # post conv
        activation_post = AliasFreeActivation1d(activation=activation_class(ch, alpha_logscale=snake_logscale))
        conv_post = WeightNormConv1D(in_channels=ch, out_channels=1, kernel_size=7)

        self.generator = InputUpSamplingResidualWaveformGenerator(pre_convolution=pre_convolution,
                                                                  upsampling_modules=ups,
                                                                  resblocks=resblocks,
                                                                  post_activation=activation_post,
                                                                  post_convolution=conv_post)

    def forward(self, x):
        return self.generator(x)


class BigVGANMRD(torch.nn.Module):
    def __init__(self,
                 n_ffts=(1024, 2048, 512),
                 hop_lengths=(120, 240, 50),
                 win_lengths=(600, 1200, 240)):
        super().__init__()
        res_discriminators = nn.ModuleList([ResolutionDiscriminator(n_fft=n_fft, hop_length=hop_len, win_length=win_len)
                                            for n_fft, hop_len, win_len in zip(n_ffts, hop_lengths, win_lengths)])
        self.multi_resolution_discriminator = MultiResolutionDiscriminator(res_discriminators)

    def forward(self, x):
        return self.multi_resolution_discriminator(x)


class BigVGANMPD(torch.nn.Module):
    def __init__(self, mpd_reshapes=(2, 3, 5, 7, 11)):
        super().__init__()
        period_discriminators = nn.ModuleList([PeriodDiscriminator(period=rs) for rs in mpd_reshapes])
        self.multi_resolution_discriminator = MultiResolutionDiscriminator(period_discriminators)

    def forward(self, x):
        return self.multi_resolution_discriminator(x)


# ---- below are other commonly used configurations with pre-trained versions of the model from NVIDIA ---- #


class BigVGANBase(BigVGAN):
    def __init__(self, **kwargs):
        super().__init__(upsample_rates=(8, 8, 2, 2),
                         upsample_kernel_sizes=(16, 16, 4, 4),
                         upsample_initial_channel=512,
                         **kwargs)


class BigVGAN100Bands(BigVGAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BigVGANBase100Bands(BigVGANBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BigVGAN80Bands(BigVGAN):
    def __init__(self, **kwargs):
        super().__init__(input_channel_size=80,
                         **kwargs)


class BigVGANBase80Bands(BigVGANBase):
    def __init__(self, **kwargs):
        super().__init__(input_channel_size=80,
                         **kwargs)
