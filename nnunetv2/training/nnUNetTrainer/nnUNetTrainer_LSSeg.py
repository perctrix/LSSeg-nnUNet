from typing import override
from mipcandy import LayerT
import torch.nn as nn
from nnunetv2.network_architecture.lsnet_seg import build_lsnet_seg_d3
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_LSSeg(nnUNetTrainer):
    @override
    def initialize(self):
        super().initialize()
        self.network = build_lsnet_seg_d3(in_ch=self.num_input_channels, variant='b', num_classes=4,
                                          conv=LayerT(nn.Conv3d), norm=LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                                          enable_deep_supervision=self.enable_deep_supervision,)