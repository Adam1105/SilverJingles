import torch.nn
from torch.nn.utils import remove_weight_norm

from silver_jingles.utils.weights_utils import init_weights


class InputUpSamplingResidualWaveformGenerator(torch.nn.Module):
    def __init__(self,
                 pre_convolution: torch.nn.Module,
                 upsampling_modules: torch.nn.ModuleList,
                 resblocks: torch.nn.ModuleList,
                 post_activation: torch.nn.Module,
                 post_convolution: torch.nn.Module):
        super().__init__()

        self.conv_pre = pre_convolution
        self.ups = upsampling_modules
        self.resblocks = resblocks
        self.activation_post = post_activation
        self.conv_post = post_convolution


        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # up-sampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # resblocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
