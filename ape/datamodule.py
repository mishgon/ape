from dataclasses import dataclass
from typing import Tuple, Optional, Union, Sequence
from pathlib import Path
import random
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from imops import crop_to_box, zoom

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from .utils import load_numpy, load_json, normalize_axis_list


@dataclass
class PreparedDataDirs:
    nlst: str
    amos_ct_labeled_train: str
    amos_ct_unlabeled_train: str
    amos_ct_val: str
    abdomen_atlas: str
    flare23_labeled_train: str
    flare23_unlabeled_train: str
    flare23_labeled_val: str
    lidc: str
    midrc_ricord_1a: str


@dataclass
class SpatialAugmentations:
    min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5)
    max_voxel_spacing: Tuple[float, float, float] = (4.0, 4.0, 6.0)
    crop_size: Tuple[int, int, int] = (96, 96, 64)


@dataclass
class ColorAugmentations:
    blur_or_sharpen_p: float = 0.8
    blur_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_alpha_range: Tuple[float, float] = (0.0, 2.0)
    noise_p: float = 0.8
    noise_sigma_range: float = (0.0, 0.1)
    invert_p: float = 0.0
    brightness_p: float = 0.8
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_p: float = 0.8
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gamma_p: float = 0.8
    gamma_range: Tuple[float, float] = (0.8, 1.25)


@dataclass
class Masking:
    p: float = 0.5
    ratio: float = 0.6
    block_size: Tuple[int, int, int] = (24, 24, 16)


class APEDataModule(pl.LightningDataModule):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int = 1000,
            spatial_augmentations: SpatialAugmentations = SpatialAugmentations(),
            color_augmentations: ColorAugmentations = ColorAugmentations(),
            masking: Masking = Masking(),
            num_crops_per_image: int = 8,
            num_voxels_per_crop: int = 1024,
            num_background_voxels_per_crop: int = 1024,
            num_images_per_epoch: int = 3000,
            num_workers: int = 0,
            prefetch_factor: Optional[int] = None,
            random_seed: int = 42
    ) -> None:
        super().__init__()

        self.prepared_data_dirs = prepared_data_dirs
        self.nlst_val_size = nlst_val_size
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.masking = masking
        self.num_crops_per_image = num_crops_per_image
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_background_voxels_per_crop = num_background_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = 'fit') -> None:
        self.train_dataset = _APEDataset(
            prepared_data_dirs=self.prepared_data_dirs,
            nlst_val_size=self.nlst_val_size,
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            masking=self.masking,
            num_crops_per_image=self.num_crops_per_image,
            num_voxels_per_crop=self.num_voxels_per_crop,
            num_background_voxels_per_crop=self.num_background_voxels_per_crop,
            num_images_per_epoch=self.num_images_per_epoch,
            random_seed=self.random_seed
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )


class _APEDataset(Dataset):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int,
            spatial_augmentations: SpatialAugmentations,
            color_augmentations: ColorAugmentations,
            masking: Masking,
            num_crops_per_image: int,
            num_background_voxels_per_crop: int,
            num_voxels_per_crop: int,
            num_images_per_epoch: int,
            random_seed: int
    ) -> None:
        super().__init__()

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.masking = masking
        self.num_crops_per_image = num_crops_per_image
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_background_voxels_per_crop = num_background_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch

        nlst_image_dirpaths, _ = train_test_split(sorted(Path(prepared_data_dirs.nlst).iterdir()),
                                                  test_size=nlst_val_size, random_state=random_seed)
        self.image_dirpaths = (
            nlst_image_dirpaths
            + sorted(Path(prepared_data_dirs.amos_ct_labeled_train).iterdir())
            + sorted(Path(prepared_data_dirs.amos_ct_unlabeled_train).iterdir())
            + sorted(Path(prepared_data_dirs.abdomen_atlas).iterdir())
            # + sorted(Path(prepared_data_dirs.flare23_labeled_train).iterdir())
            # + sorted(Path(prepared_data_dirs.flare23_unlabeled_train).iterdir())
        )

    def __len__(self):
        return self.num_images_per_epoch

    def __getitem__(self, index: int):
        image_dirpath = random.choice(self.image_dirpaths)
        image = load_numpy(image_dirpath / 'image.npy.gz', decompress=True)
        voxel_spacing = load_json(image_dirpath / 'voxel_spacing.json')
        body_mask = load_numpy(image_dirpath / 'body_mask.npy.gz', decompress=True)

        image = np.float32(image)

        (images_1, masks_1, voxel_indices_1, background_voxel_indices_1,
         images_2, masks_2, voxel_indices_2, background_voxel_indices_2,
         voxel_positions, background_voxel_positions_1, background_voxel_positions_2) = zip(*[
             _get_augmented_crops(
                image=image,
                voxel_spacing=voxel_spacing,
                body_mask=body_mask,
                spatial_augmentations=self.spatial_augmentations,
                color_augmentations=self.color_augmentations,
                masking=self.masking,
                num_voxels_per_crop=self.num_voxels_per_crop,
                num_background_voxels_per_crop=self.num_background_voxels_per_crop
            )
            for _ in range(self.num_crops_per_image)
        ])

        return (
            torch.from_numpy(np.stack(images_1)),
            torch.from_numpy(np.stack(masks_1)),
            list(map(torch.from_numpy, voxel_indices_1)),
            list(map(torch.from_numpy, background_voxel_indices_1)),
            torch.from_numpy(np.stack(images_2)),
            torch.from_numpy(np.stack(masks_2)),
            list(map(torch.from_numpy, voxel_indices_2)),
            list(map(torch.from_numpy, background_voxel_indices_2)),
            list(map(torch.from_numpy, voxel_positions)),
            list(map(torch.from_numpy, background_voxel_positions_1)),
            list(map(torch.from_numpy, background_voxel_positions_2))
        )


def _get_augmented_crops(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        body_mask: np.ndarray,
        spatial_augmentations: SpatialAugmentations,
        color_augmentations: ColorAugmentations,
        masking: Masking,
        num_voxels_per_crop: int,
        num_background_voxels_per_crop: int
) -> Tuple:
    image_size = np.array(image.shape, dtype='int64')
    voxel_spacing = np.array(voxel_spacing, dtype='float32')
    crop_size = np.array(spatial_augmentations.crop_size, dtype='int64')
    min_voxel_spacing = np.array(spatial_augmentations.min_voxel_spacing, dtype='float32')
    max_voxel_spacing = np.array(spatial_augmentations.max_voxel_spacing, dtype='float32')
    max_voxel_spacing = np.minimum(max_voxel_spacing, voxel_spacing * image_size / crop_size)

    voxel_spacing_1 = np.random.uniform(min_voxel_spacing, max_voxel_spacing)
    voxel_spacing_2 = np.random.uniform(min_voxel_spacing, max_voxel_spacing)

    crop_size_before_resize_1 = np.int64(np.round(crop_size * voxel_spacing_1 / voxel_spacing))
    crop_size_before_resize_2 = np.int64(np.round(crop_size * voxel_spacing_2 / voxel_spacing))

    resize_factor_1 = crop_size / crop_size_before_resize_1
    resize_factor_2 = crop_size / crop_size_before_resize_2

    voxel_index = np.random.randint(0, image_size, size=(1, 3))
    crop_box_1 = _get_random_box(image_size, crop_size_before_resize_1, voxel_index)
    crop_box_2 = _get_random_box(image_size, crop_size_before_resize_2, voxel_index)
    overlap_box = _get_overlap_box(crop_box_1, crop_box_2)
    voxel_indices = overlap_box[0] + np.argwhere(crop_to_box(body_mask, overlap_box))
    if len(voxel_indices) > num_voxels_per_crop:
        voxel_indices = voxel_indices[np.random.choice(len(voxel_indices), num_voxels_per_crop, replace=False)]

    voxel_positions = np.float32(voxel_indices) * voxel_spacing

    image_1, mask_1, voxel_indices_1, background_voxel_indices_1, background_voxel_positions_1 = _get_augmented_crop(
        image, voxel_spacing, body_mask, voxel_indices, crop_box_1, resize_factor_1,
        color_augmentations, masking, num_background_voxels_per_crop
    )
    image_2, mask_2, voxel_indices_2, background_voxel_indices_2, background_voxel_positions_2 = _get_augmented_crop(
        image, voxel_spacing, body_mask, voxel_indices, crop_box_2, resize_factor_2,
        color_augmentations, masking, num_background_voxels_per_crop
    )
    return (image_1, mask_1, voxel_indices_1, background_voxel_indices_1,
            image_2, mask_2, voxel_indices_2, background_voxel_indices_2,
            voxel_positions, background_voxel_positions_1, background_voxel_positions_2)


def _get_augmented_crop(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        body_mask: np.ndarray,
        voxel_indices: np.ndarray,
        crop_box: np.ndarray,
        resize_factor: np.ndarray,
        color_augmentations: ColorAugmentations,
        masking: Masking,
        num_background_voxels_per_crop: int
) -> Tuple:
    image = crop_to_box(image, crop_box)
    voxel_indices = voxel_indices - crop_box[0]

    background_voxel_indices = np.argwhere(~crop_to_box(body_mask, crop_box))
    if len(background_voxel_indices) > num_background_voxels_per_crop:
        random_subset = np.random.choice(len(background_voxel_indices), num_background_voxels_per_crop, replace=False)
        background_voxel_indices = background_voxel_indices[random_subset]
    background_voxel_positions = np.float32(crop_box[0] + background_voxel_indices) * voxel_spacing

    image = zoom(np.ascontiguousarray(image), resize_factor, backend='Scipy')
    voxel_indices = np.int64(np.floor(voxel_indices * resize_factor))
    background_voxel_indices = np.int64(np.floor(background_voxel_indices * resize_factor))
    voxel_spacing = voxel_spacing / resize_factor

    # augment colors
    image = _augment_color(image, voxel_spacing, color_augmentations)

    # sample mask
    mask = _get_random_mask(image.shape, masking)

    # add channel dim
    image = np.expand_dims(image, axis=0)

    return image, mask, voxel_indices, background_voxel_indices, background_voxel_positions


def _augment_color(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        color_augmentations: ColorAugmentations
) -> np.ndarray:
    if random.uniform(0, 1) < color_augmentations.blur_or_sharpen_p:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(*color_augmentations.blur_sigma_range) / voxel_spacing[:2]
            image = _gaussian_filter(image, sigma, axis=(0, 1))
        else:
            sigma = random.uniform(*color_augmentations.sharpen_sigma_range) / voxel_spacing[:2]
            alpha = random.uniform(*color_augmentations.sharpen_alpha_range)
            image = _gaussian_sharpen(image, sigma, alpha, axis=(0, 1))

    if random.uniform(0, 1) < color_augmentations.noise_p:
        # gaussian noise
        noise_sigma = random.uniform(*color_augmentations.noise_sigma_range)
        image = image + np.float32(np.random.normal(0, noise_sigma, size=image.shape))

    if random.uniform(0, 1) < color_augmentations.invert_p:
        # invert
        image = 1.0 - image

    if random.uniform(0, 1) < color_augmentations.brightness_p:
        # adjust brightness
        brightness_factor = random.uniform(*color_augmentations.brightness_range)
        image = np.clip(image * brightness_factor, 0.0, 1.0)

    if random.uniform(0, 1) < color_augmentations.contrast_p:
        # adjust contrast
        contrast_factor = random.uniform(*color_augmentations.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast_factor + mean, 0.0, 1.0)

    if random.uniform(0, 1) < color_augmentations.gamma_p:
        image = np.clip(image, 0.0, 1.0)
        gamma = random.uniform(*color_augmentations.gamma_range)
        image = np.power(image, gamma)

    return image


def _get_random_mask(size: Sequence[int], masking: Masking) -> np.ndarray:
    if masking.ratio == 0.0 or random.uniform(0, 1) > masking.p:
        return np.ones(size, dtype='float32')

    size = np.array(size, dtype='int64')
    block_size = np.array(masking.block_size, dtype='int64')

    assert np.all(size % block_size == 0)

    mask = np.ones(size // block_size, dtype='float32')
    mask[np.unravel_index(np.random.permutation(mask.size)[:int(mask.size * masking.ratio)], mask.shape)] = 0.0
    assert (mask != 1.0).any()
    for axis, repeats in enumerate(block_size):
        mask = np.repeat(mask, repeats, axis)

    return mask


def _gaussian_filter(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    axis = normalize_axis_list(axis, x.ndim)
    sigma = np.broadcast_to(sigma, len(axis))
    for sgm, ax in zip(sigma, axis):
        x = gaussian_filter1d(x, sgm, ax)
    return x


def _gaussian_sharpen(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        alpha: float,
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    return x + alpha * (x - _gaussian_filter(x, sigma, axis))


def _get_random_box(
        image_size: Sequence[int],
        box_size: Sequence[int],
        pins: Optional[np.ndarray] = None
) -> np.ndarray:
    image_size = np.array(image_size)
    box_size = np.array(box_size)
    if not np.all(image_size >= box_size):
        raise ValueError(f'Can\'t sample patch of size {box_size} from image of size {image_size}')

    min_start = 0
    max_start = image_size - box_size

    if pins is not None:
        assert pins.ndim == 2
        assert pins.shape[1] == 3

        min_start = np.maximum(min_start, np.max(pins, axis=0) - box_size + 1)
        max_start = np.minimum(max_start, np.min(pins, axis=0))

        assert np.all(min_start <= max_start)

    start = np.random.randint(min_start, max_start + 1)

    return np.array([start, start + box_size])


def _get_overlap_box(*boxes: np.ndarray) -> np.ndarray:
    start = np.max([box[0] for box in boxes], axis=0)
    stop = np.min([box[1] for box in boxes], axis=0)
    if not np.all(start < stop):
        return
    return np.array([start, stop])
