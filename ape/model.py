from typing import Optional, Union, Tuple, Literal, Sequence
import itertools
import math

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from medimm.fpn_3d import FPN3d, FPNLinearDenseHead3d


class APE(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.fpn = FPN3d(
            in_channels=1,
            stem_stride=4,
            out_channels=(96, 192, 384, 768),
            hidden_factors=4.0,
            depths=((3, 1), (3, 1), (9, 1), 3),
            stem_kernel_size=4,
            stem_padding=0,
            drop_path_rate=0.0,
            final_ln=True,
            final_affine=True,
            final_gelu=True,
            mask_token=True
        )
        self.head = FPNLinearDenseHead3d(
            out_channels=3,
            fpn_stem_stride=self.fpn.stem_stride,
            fpn_out_channels=self.fpn.out_channels
        )
        self.final_act = nn.Sigmoid()

        if pretrained:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(
                repo_id='mishgon/ape',
                filename=f'ape.pt',
                revision='a743d914ca8ba47d4e7371c6906c40c976437685'
            )
            self.load_state_dict(torch.load(weights_path))

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.final_act(self.head(image, self.fpn(image, mask), upsample=False))

    @staticmethod
    def to_sin_cos(ape_maps: torch.Tensor, embed_dim: int = 24) -> torch.Tensor:
        """Computes sin-cos anatomical positional embeddings.

        Args:
            ape_maps (torch.Tensor):
                Tensor of size ``(n, 3, h, w, d)``.
            embed_dim (int, optional):
                Defaults to 24.

        Returns:
            torch.Tensor:
                Tensor of size ``(n, embed_dim, h, w, d)``.
        """
        assert ape_maps.shape[1] == 3
        assert embed_dim % 6 == 0

        frequencies = 2 ** torch.arange(-1, embed_dim // 6 - 1).to(ape_maps)
        ape_maps = 2 * math.pi * ape_maps.unsqueeze(-1) * frequencies
        ape_maps = torch.cat([ape_maps.sin(), ape_maps.cos()], dim=-1)
        ape_maps = ape_maps.movedim(-1, 1).flatten(1, 2)

        assert ape_maps.shape[1] == embed_dim

        return ape_maps


class APELightningModule(pl.LightningModule):
    def __init__(
            self,
            i_weight: float = 1.0,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.ape = APE(pretrained=False)
        self.scale = nn.Parameter(torch.ones(3, 1, 1, 1))
        self.i_weight = i_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        (images_1, masks_1, voxel_indices_1, background_voxel_indices_1,
         images_2, masks_2, voxel_indices_2, background_voxel_indices_2,
         voxel_positions, background_voxel_positions_1, background_voxel_positions_2) = batch

        ape_maps_1 = self.ape(images_1, masks_1) * self.scale
        ape_maps_2 = self.ape(images_2, masks_2) * self.scale
        stride = self.ape.fpn.stem_stride

        embeds_1 = batched_take_embeds_from_map(ape_maps_1, voxel_indices_1, stride)  # (N, 3)
        embeds_2 = batched_take_embeds_from_map(ape_maps_2, voxel_indices_2, stride)  # (N, 3)

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'ape/i_reg', i_reg, on_epoch=True, on_step=True)

        background_embeds_1 = batched_take_embeds_from_map(ape_maps_1, background_voxel_indices_1, stride)
        background_embeds_2 = batched_take_embeds_from_map(ape_maps_2, background_voxel_indices_2, stride)

        embeds_1 = torch.cat([embeds_1, background_embeds_1])
        embeds_2 = torch.cat([embeds_2, background_embeds_2])

        pos_1 = torch.cat(voxel_positions + background_voxel_positions_1)
        pos_2 = torch.cat(voxel_positions + background_voxel_positions_2)
        pos_mean = pos_1.mean(dim=0)
        pos_std = pos_1.std(dim=0)
        pos_1 = (pos_1 - pos_mean) / pos_std
        pos_2 = (pos_2 - pos_mean) / pos_std

        embeds_pdist = torch.norm(embeds_1.unsqueeze(1) - embeds_2, dim=-1)  # (N, N)
        pdist_mm = torch.norm(pos_1.unsqueeze(1) - pos_2, dim=-1)  # (N, N)
        pdist_reg = F.mse_loss(embeds_pdist, pdist_mm)
        self.log(f'ape/pdist_reg', pdist_reg, on_epoch=True, on_step=True)

        loss = pdist_reg + self.i_weight * i_reg
        self.log(f'ape/loss', loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            eps=1e-3,
            weight_decay=self.weight_decay,
        )

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

    def transfer_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_to_device(batch, device, dataloader_idx)


def take_embeds_from_map(
        embed_map: torch.Tensor,
        voxel_indices: torch.Tensor,
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    if stride == 1:
        return embed_map.movedim(0, -1)[voxel_indices.unbind(1)]

    stride = torch.tensor(stride).to(voxel_indices)
    min_indices = torch.tensor(0).to(voxel_indices)
    max_indices = torch.tensor(embed_map.shape[-3:]).to(voxel_indices) - 1
    if mode == 'nearest':
        indices = voxel_indices // stride
        indices = torch.clamp(indices, min_indices, max_indices)
        return embed_map.movedim(0, -1)[indices.unbind(1)]
    elif mode == 'trilinear':
        x = embed_map.movedim(0, -1)
        points = (voxel_indices + 0.5) / stride - 0.5
        starts = torch.floor(points).long()  # (n, 3)
        stops = starts + 1  # (n, 3)
        embeds = 0.0
        for mask in itertools.product((0, 1), repeat=3):
            mask = torch.tensor(mask, device=voxel_indices.device, dtype=bool)
            corners = torch.where(mask, starts, stops)  # (n, 3)
            corners = torch.clamp(corners, min_indices, max_indices)  # (n, 3)
            weights = torch.prod(torch.where(mask, 1 - (points - starts), 1 - (stops - points)), dim=-1, keepdim=True)  # (n, 1)
            embeds = embeds + weights.to(x) * x[corners.unbind(-1)]  # (n, d)
        return embeds
    else:
        raise ValueError(mode)


def batched_take_embeds_from_map(
        embed_maps_batch: torch.Tensor,
        voxel_indices_batch: Sequence[torch.Tensor],
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    return torch.cat([
        take_embeds_from_map(feature_map, voxel_indices, stride, mode)
        for feature_map, voxel_indices in zip(embed_maps_batch, voxel_indices_batch, strict=True)
    ])
