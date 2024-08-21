from typing import Iterable, Optional

import logging
logging.getLogger().setLevel(logging.WARNING)

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from vox2vec.nn.unet import UNet3d


class APE(nn.Module):


class APELightningModule(pl.LightningModule):
    def __init__(
            self,
            in_channels: int = 1,
            i_weight: float = 1.0,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.nn = UNet3d(
            in_channels=in_channels,
            out_channels=3,
            stem_stride=4,
            fpn_out_channels=(96, 192, 384, 768),
            fpn_hidden_factors=4.,
            fpn_depths=((2, 1), (2, 1), (8, 1), 3),
            fpn_blocks='convnext',
            stem_norm='ln',
            fpn_middle_norm='ln',
            fpn_final_norm='ln',
            fpn_final_affine=True,
            fpn_final_act='gelu',
            final_norm='none',
            final_affine=False,
            final_act='sigmoid'
        )
        self.scale = nn.Parameter(torch.ones(3, 1, 1, 1))

        self.i_weight = i_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def _take(self, ape_maps: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        min_indices = torch.tensor(0, device=self.device)
        max_indices = torch.tensor(ape_maps.shape[-3:], device=self.device) - 1
        return torch.cat([
            ape_maps[j].movedim(0, -1)[torch.clamp(v // 4, min_indices, max_indices).unbind(1)]  # stem_stride = 4
            for j, v in enumerate(voxels)
        ])

    def training_step(self, batch, batch_idx):
        assert self.nn.training

        ape_maps_1 = self.nn(batch['patches_1'], batch['masks_1']) * self.scale
        ape_maps_2 = self.nn(batch['patches_2'], batch['masks_2']) * self.scale

        embeds_1 = self._take(ape_maps_1, batch['voxels_per_patch_1'])  # (N, 3)
        embeds_2 = self._take(ape_maps_2, batch['voxels_per_patch_2'])  # (N, 3)

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'ape/i_reg', i_reg, on_epoch=True, on_step=True, sync_dist=True)

        embeds_1 = torch.cat([embeds_1, self._take(ape_maps_1, batch['background_voxels_per_patch_1'])])
        embeds_2 = torch.cat([embeds_2, self._take(ape_maps_2, batch['background_voxels_per_patch_2'])])

        pos_1 = torch.cat(batch['pos_per_patch'] + batch['background_pos_per_patch_1'])  # (N, 3)
        pos_2 = torch.cat(batch['pos_per_patch'] + batch['background_pos_per_patch_2'])
        pos_mean = pos_1.mean(dim=0)
        pos_std = pos_1.std(dim=0)
        pos_1 = (pos_1 - pos_mean) / pos_std
        pos_2 = (pos_2 - pos_mean) / pos_std

        embeds_pdist = torch.norm(embeds_1.unsqueeze(1) - embeds_2, dim=-1)  # (N, N)
        pdist_mm = torch.norm(pos_1.unsqueeze(1) - pos_2, dim=-1)  # (N, N)
        pdist_reg = F.mse_loss(embeds_pdist, pdist_mm)
        self.log(f'ape/pdist_reg', pdist_reg, on_epoch=True, on_step=True, sync_dist=True)

        ape_reg = pdist_reg + self.i_weight * i_reg
        self.log(f'ape/total_reg', ape_reg, on_epoch=True, on_step=True, sync_dist=True)

        return ape_reg

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)

        if self.warmup_steps is None:
            return optimizer
    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
