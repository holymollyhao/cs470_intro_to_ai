import copy

import torch

class Identity(torch.nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def remove_bn_resnet50(net):
    
    net = copy.deepcopy(net)
    
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    net.eval()

    net.conv1 = fuse(net.conv1, net.bn1)
    net.bn1 = Identity()

    for name, module in net.named_modules():
        # print(name, module)
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    net.train()
    return net
    # print(net.state_dict().keys())


def remove_bn_resnext29(net):
    net = copy.deepcopy(net)
    # for resnext
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    net.eval()

    net.conv_1_3x3 = fuse(net.conv_1_3x3, net.bn_1)
    net.bn_1 = Identity()

    for name, module in net.named_modules():
        dict = {
            'conv_reduce': 'bn_reduce',
            'conv_conv': 'bn',
            'conv_expand': 'bn_expand'
        }
        if name.startswith("stage") and len(name) == 7:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = dict[name2]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    net.train()
    return net

def remove_bn_wideresent(net):
    net = copy.deepcopy(net)
    # for resnext
    fuse = torch.nn.utils.fusion.fuse_linear_bn_eval
    net.eval()

    # net.conv_1_3x3 = fuse(net.conv_1_3x3, net.bn_1)
    # net.bn_1 = Identity()

    for name, module in net.named_modules():
        # print("############################################################")
        # print(name)
        # print(module)
        # dict = {
        #     'conv_reduce' : 'bn_reduce',
        #     'conv_conv' : 'bn',
        #     'conv_expand' : 'bn_expand'
        # }
        if name.startswith("block") and len(name) == 6:
            for b, bottleneck in enumerate(module.layer):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("bn"):
                        setattr(bottleneck, name2, Identity())

    net.bn1 = Identity()
    net.train()
    return net


def remove_bn_harth(net):
    net = copy.deepcopy(net)
    # for resnext
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    net.eval()

    # net.conv_1_3x3 = fuse(net.conv_1_3x3, net.bn_1)
    # net.bn_1 = Identity()
    # print(net.state_dict().keys())

    for name, module in net[0].named_modules():
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(name)
        # print(module)
        if isinstance(module, torch.nn.Sequential):
            for name2, module2 in module.named_modules():
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # print(name2)
                # print(module2)
                if isinstance(module2, torch.nn.BatchNorm1d) and int(name2):
                    setattr(module, name2, Identity())

    # for name, module in net.named_modules():
    #     print("#########################################################################")
    #     print(name)
    #     print(module)
    #     break

    net.train()
    return net
