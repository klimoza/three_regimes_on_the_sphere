## 5-Layer CNN for CIFAR (fully scale-invariant; custom init; custom weight norm applied)
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
# based on https://gitlab.com/harvard-machine-learning/double-descent/blob/master/models/mcnn.py

from torchvision import transforms
import torch.nn as nn
from .utils import WeightNorm

def block(input, output, eps):
    # Layer i
    module_list = [
        WeightNorm(nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=False), eps=eps),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ]
    return module_list

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class ConvNetDepth(nn.Module):
    def __init__(self, init_channels=64, num_classes=10, max_depth=3, init_scale=50.0, eps=1e-4, learn_bias=False, zero_bias=False):
        super(ConvNetDepth, self).__init__()
        c = init_channels
        module_list = block(3, c, eps)
        module_list = module_list[:-1]  # no max pooling at end of first layer

        current_width = c
        last_zero = max_depth // 3 + 1 * (max_depth % 3 > 0) - 1
        for i in range(max_depth // 3 + 1 * (max_depth % 3 > 0)):
            if i != last_zero:
                module_list.extend(block(current_width, current_width, eps))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width, eps))
                current_width = 2 * current_width

        last_one = max_depth // 3 + 1 * (max_depth % 3 > 1) - 1
        for i in range(max_depth // 3 + 1 * (max_depth % 3 > 1)):
            if i != last_one:
                module_list.extend(block(current_width, current_width, eps))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width, eps))
                current_width = 2 * current_width

        last_two = max_depth // 3 + 1 * (max_depth % 3 > 2) - 1
        for i in range(max_depth // 3 + 1 * (max_depth % 3 > 2)):
            if i != last_two:
                module_list.extend(block(current_width, current_width, eps))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width, eps))
                current_width = 2 * current_width

        pooling_increaser = 1
        if max_depth < 3:
            pooling_increaser = (3 - max_depth) * 2

        linear_layer = [
            nn.MaxPool2d(4 * pooling_increaser),
            Flatten(),
            nn.Linear(current_width, num_classes, bias=True)
        ]

        # Custom initialization: just set the norm higher
        alpha = init_scale
        W = linear_layer[-1].weight.data
        linear_layer[-1].weight.data = alpha * W / W.norm()

        # Freeze the parameters of the last FC layer for SI of the objective
        for n, param in linear_layer[-1].named_parameters():
            if learn_bias and "bias" in n:
                continue
            param.requires_grad = False

        # Freeze weight_g component of the parameters for SI of the objective
        for layer in module_list:
            for n, param in layer.named_parameters():
                if "weight_g" in n:
                    param.requires_grad = False

        if zero_bias:
            linear_layer[-1].bias.data.zero_()

        self.conv_layers = nn.Sequential(*module_list)
        self.linear_layers = nn.Sequential(*linear_layer)

    def forward(self, x):
        return self.linear_layers(self.conv_layers(x))

class ConvNetSICI3WN2:
    base = ConvNetDepth
    args = []
    kwargs = {}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
