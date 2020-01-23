from .rcan_structure import Rcan
import torch.nn as nn

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=1, act_fun='LeakyReLU',
            skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride',
            n_resgroups=5, n_resblocks=10, n_feats=64, reduction=16):
    if NET_TYPE == 'rcan':
        net = Rcan(n_resgroups, n_resblocks, n_feats, reduction)

    else:
        assert False

    return net