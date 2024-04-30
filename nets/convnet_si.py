## 5-Layer CNN for CIFAR
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
# based on https://gitlab.com/harvard-machine-learning/double-descent/blob/master/models/mcnn.py

from torchvision import transforms

import torch.nn as nn
import torch
from training_utils import unflatten_like

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

def block(input, output):
    # Layer i
    module_list = [
        nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=False),  # bias is needless befor BN
        nn.BatchNorm2d(output, affine=False),#, eps=0.0),  # affine is False for clarity of SI experiments
        nn.ReLU(),
        nn.MaxPool2d(2)
    ]
    return module_list

def check_si_name(n, model_name='ResNet18'):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    return False

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class ConvNetDepth(nn.Module):
    def __init__(self, init_channels=64, num_classes=10, max_depth=3, init_scale=-1,
                 su_init=False):
        super(ConvNetDepth, self).__init__()
        c = init_channels
        module_list = block(3, c)
        module_list = module_list[:-1]  # no max pooling at end of first layer

        current_width = c
        last_zero = max_depth // 3 + 1 * (max_depth % 3 > 0) - 1
        for i in range(max_depth // 3 + 1 * (max_depth % 3 > 0)):
            if i != last_zero:
                module_list.extend(block(current_width, current_width))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width))
                current_width = 2 * current_width

        last_one = max_depth // 3 + 1 * (max_depth % 3 > 1) - 1
        for i in range(max_depth // 3 + 1 * (max_depth % 3 > 1)):
            if i != last_one:
                module_list.extend(block(current_width, current_width))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width))
                current_width = 2 * current_width

        last_two = max_depth // 3 + 1 * (max_depth % 3 > 2) - 1
        for i in range(max_depth // 3 + 1 * (max_depth % 3 > 2)):
            if i != last_two:
                module_list.extend(block(current_width, current_width))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width))
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
        if init_scale >0:
            alpha = init_scale
            W = linear_layer[-1].weight.data
            linear_layer[-1].weight.data = alpha * W / W.norm()
        
        # Freeze the parameters in the last FC layer for SI of the objective
        # for param in linear_layer[-1].parameters():
        #     param.requires_grad = False

        # module_list.extend(linear_layer)

        self.conv_layers = nn.Sequential(*module_list)
        self.linear_layers = nn.Sequential(*linear_layer)
        
        if su_init:
            self._su_init()

    def _su_init(self):
        params = [p for n, p in self.named_parameters() if check_si_name(n, 'ConvNetSI')]  # BN-ed params
        N = sum(p.numel() for p in params)
        pnorm = sum((p ** 2).sum() for p in params) ** 0.5

        vec = torch.randn(N, device=params[0].device)
        vec /= torch.norm(vec)
        vec *= pnorm

        tensors = unflatten_like(vec, params)
        for param, tensor in zip(params, tensors):
            param.data = tensor
        
    def forward(self, x):
        return self.linear_layers(self.conv_layers(x))

class ConvNetSI:
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
