from torch import nn
from torch.nn.utils import remove_weight_norm
from silver_jingles.utils.weights_utils import init_weights


class ResBlock1(torch.nn.Module):
    def __init__(self,
                 convs1: nn.ModuleList,
                 activations1: nn.ModuleList,
                 convs2: nn.ModuleList,
                 activations2: nn.ModuleList):
        super().__init__()
        if not (len(convs1) == len(convs2) == len(activations1) == len(activations2)):
            raise RuntimeError(f"ResBlock1 assumes the same length for modules lists of:"
                               f" convs1, convs2, activations1, activations2. While provided respectively are: "
                               f"{len(convs1), len(convs2), len(activations1), len(activations2)}")
        self.convs1 = convs1
        self.convs2 = convs2
        self.activations1 = activations1
        self.activations2 = activations2

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, a1, c2, a2 in zip(self.convs1, self.activations1, self.convs2, self.activations2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
