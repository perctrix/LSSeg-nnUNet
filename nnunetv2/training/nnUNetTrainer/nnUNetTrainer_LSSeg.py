from typing import override, Union, List, Tuple, Optional
from mipcandy import LayerT
import torch
import torch.nn as nn
from torch._dynamo import OptimizedModule
from nnunetv2.network_architecture.lsnet_seg import build_lsnet_seg_d3
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

class nnUNetTrainer_LSSeg(nnUNetTrainer):
    """
    nnUNet Trainer for LSNetSegD3 architecture.

    Key differences from standard nnUNet:
    - Uses LSNetSegD3 instead of PlainConvUNet
    - Deep supervision outputs are ALL at full resolution (not downsampled)
    - 4 outputs: [main, ds1, ds2, ds3], all at same resolution
    """
    @staticmethod
    @override
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build LSNetSegD3 network instead of the default nnUNet architecture.
        """
        return build_lsnet_seg_d3(
            in_ch=num_input_channels,
            num_classes=num_output_channels,
            variant='b',
            num_dims=3,
            conv=LayerT(nn.Conv3d),
            norm=LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
            enable_deep_supervision=enable_deep_supervision,
        )

    @override
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Override for LSNetSegD3 which uses enable_deep_supervision attribute
        instead of decoder.deep_supervision
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.enable_deep_supervision = enabled

    @override
    def _get_deep_supervision_scales(self) -> Optional[List[List[float]]]:
        """
        Override for LSNetSegD3 which outputs all deep supervision at full resolution.

        LSNetSegD3 has 4 outputs when deep_supervision is enabled:
        [main_out, ds1, ds2, ds3] - ALL at full resolution (1.0, 1.0, 1.0 for 3D)

        Standard nnUNet expects downsampled outputs, but LSNetSegD3 upsamples
        all deep supervision outputs to full resolution, so we return
        scales of [1.0, 1.0, 1.0] for each output.
        """
        if self.enable_deep_supervision:
            # 4 outputs, all at full resolution for 3D
            return [[1.0, 1.0, 1.0]] * 4
        else:
            return None

    def _remap_labels(self, target):
        """
        Remap BRaTS labels from [0, 1, 2, 4] to continuous [0, 1, 2, 3].
        This is needed because the dataset was preprocessed without label remapping.

        Mapping:
        - 0 -> 0 (background)
        - 1 -> 1 (necrotic tumor core)
        - 2 -> 2 (peritumoral edema)
        - 4 -> 3 (enhancing tumor)
        """
        if isinstance(target, list):
            return [self._remap_labels(t) for t in target]
        # Remap label 4 to 3
        target = torch.where(target == 4, torch.tensor(3, device=target.device, dtype=target.dtype), target)
        return target

    @override
    def train_step(self, batch: dict) -> dict:
        """
        Override train_step to remap labels before computing loss.
        """
        data = batch['data']
        target = batch['target']

        # Remap labels from [0,1,2,4] to [0,1,2,3]
        target = self._remap_labels(target)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    @override
    def validation_step(self, batch: dict) -> dict:
        """
        Override validation_step to remap labels before computing loss.
        """
        data = batch['data']
        target = batch['target']

        # Remap labels from [0,1,2,4] to [0,1,2,3]
        target = self._remap_labels(target)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data

            # Handle deep supervision loss
            # In eval mode, network returns single tensor but target is still a list
            if isinstance(target, list) and not isinstance(output, list):
                # Use only the first (full resolution) target for non-deep-supervision output
                target_for_loss = [target[0]]
                output_for_loss = [output]
                l = self.loss(output_for_loss, target_for_loss)
                target = target[0]  # Use full resolution target for metrics
            else:
                l = self.loss(output, target)
                if isinstance(target, list):
                    target = target[0]  # Use full resolution target for metrics

            if isinstance(output, list):
                output = output[0]  # Use full resolution output for metrics

        # compute metrics for foreground classes
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        # Create one-hot encoding for remapped labels (0,1,2,3 -> 4 channels)
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]  # remove background
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
