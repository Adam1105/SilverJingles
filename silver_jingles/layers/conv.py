import torch.nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm


class WeightNormConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._conv = weight_norm(Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        dilation=dilation, padding=self.get_padding()))

    def get_padding(self):
        return int((self._kernel_size * self._dilation - self._dilation) / 2)

    # TODO: double check if remove_weight_norm works. If so check this also on the full model


class WeightNormConvTransposed1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._conv = weight_norm(ConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                                 padding=self.get_padding()))

    def get_padding(self):
        return (self._kernel_size - self._stride) // 2

    # TODO: double check if remove_weight_norm works. If so check this also on the full model
