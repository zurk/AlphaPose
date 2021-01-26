# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
from collections import namedtuple

import torch
import torch.nn as nn

from .builder import SPPE
from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet

FastPoseOut = namedtuple("FastPoseOut", ("joints_map", "joints_radius"))

@SPPE.register_module
class FastPose(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if 'CONV_DIM' in cfg.keys():
            self.conv_dim = cfg['CONV_DIM']
        else:
            self.conv_dim = 128
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = SEResnet(
                f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm   # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if self.conv_dim == 256:
            self.duc2 = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)
        self.conv_out_radius = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

        inputSize = self._preset_cfg['HEATMAP_SIZE'][0] * self._preset_cfg['HEATMAP_SIZE'][1] * self._preset_cfg['NUM_JOINTS']
        self.linear_radius = nn.Linear(inputSize, self._preset_cfg['NUM_JOINTS'])

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out_joints = self.conv_out(out)

        out_radius = self.conv_out_radius(out)
        out_radius = torch.reshape(out_radius, shape=(out_radius.shape[0], -1))
        out_radius = self.linear_radius(out_radius)

        return FastPoseOut(out_joints, out_radius)

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.conv_out_radius.weight, std=0.001)
        nn.init.constant_(self.conv_out_radius.bias, 0)

        nn.init.xavier_uniform_(self.linear_radius.weight)
        nn.init.constant_(self.linear_radius.bias, 0.01)
