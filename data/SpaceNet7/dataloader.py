"""
spacenet7 dataloader for urban building segmentation.
supports optional hdf5 caches to accelerate data loading.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset, get_worker_info

from .preprocessing import extract_patches, load_temporal_sequence

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None


def get_dataloader(
    paths_file,
    root_dir,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    return_paths: bool = False,
    my_collate=None,
    cache_path: Optional[str] = None,
    use_hdf5: bool = False,
    preload: bool = False,
):
    """
    create dataloader for spacenet7 dataset.

    additional args:
        cache_path: optional path to hdf5 cache file
        use_hdf5:   if true, load samples from hdf5 cache (requires h5py)
        preload:    if true and not using hdf5, keep patch tensors in memory
    """
    dataset = SpaceNet7Dataset(
        csv_file=paths_file,
        root_dir=root_dir,
        transform=transform,
        return_paths=return_paths,
        patch_size=64,
        max_seq_len=24,
        use_bands=[0, 1, 2],
        cache_dir=None,
        preload=preload,
        cache_path=cache_path,
        use_hdf5=use_hdf5,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate,
    )
    return dataloader


class SpaceNet7Dataset(Dataset):
    """
    spacenet7 dataset for temporal urban building segmentation.

    loads multi-temporal satellite images and building footprint labels.
    can source data from on-the-fly AOI parsing or a pre-built hdf5 cache.
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        return_paths: bool = False,
        patch_size: int = 64,
        max_seq_len: int = 24,
        use_bands: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,  # legacy parameter (ignored)
        preload: bool = False,
        cache_path: Optional[str] = None,
        use_hdf5: bool = False,
    ):
        if isinstance(csv_file, str):
            self.aoi_list = pd.read_csv(csv_file, header=None)[0].tolist()
        elif isinstance(csv_file, (list, tuple)):
            self.aoi_list = []
            for csv in csv_file:
                self.aoi_list.extend(pd.read_csv(csv, header=None)[0].tolist())
        else:
            raise ValueError("csv_file must be str, list, or tuple")

        self.root_dir = Path(root_dir) / "train"
        self.transform = transform
        self.return_paths = return_paths
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.use_bands = list(use_bands) if use_bands is not None else [0, 1, 2]
        self.use_hdf5 = use_hdf5
        self.cache_path = Path(cache_path) if cache_path else None

        self._worker_handles: Dict[int, "h5py.File"] = {}
        self._main_handle: Optional["h5py.File"] = None
        self._aoi_names_cache: Optional[np.ndarray] = None
        self._positions_cache: Optional[np.ndarray] = None

        if self.use_hdf5:
            if h5py is None:
                raise ImportError("h5py is required to use the SpaceNet7 HDF5 cache.")
            if not self.cache_path or not self.cache_path.exists():
                raise FileNotFoundError("cache_path does not exist; run build_spacenet7_cache.py first.")
            self._init_from_hdf5()
        else:
            self.cache_dir = Path(cache_dir) if cache_dir else None
            print(f"loading spacenet7 dataset from {len(self.aoi_list)} aois...")
            self.patches = self._extract_all_patches(preload=preload)
            self.num_patches = len(self.patches)
            print(f"total patches: {self.num_patches}")

    # ------------------------------------------------------------------ #
    # hdf5 helpers
    def _init_from_hdf5(self) -> None:
        with h5py.File(self.cache_path, "r") as handle:
            self.num_patches = handle["images"].shape[0]
            self.max_seq_len = int(handle.attrs.get("max_seq_len", self.max_seq_len))
            self.patch_size = int(handle.attrs.get("patch_size", self.patch_size))
            bands = handle.attrs.get("bands")
            if bands is not None:
                self.use_bands = list(np.asarray(bands, dtype=int))
            # cache metadata for fast retrieval
            raw_names = handle["aoi_name"][:]
            self._aoi_names_cache = np.array([name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names])
            self._positions_cache = handle["top_left"][:]

    def _get_h5_file(self) -> "h5py.File":
        worker_info = get_worker_info()
        if worker_info is None:
            if self._main_handle is None:
                self._main_handle = h5py.File(self.cache_path, "r")
            return self._main_handle

        worker_id = worker_info.id
        handle = self._worker_handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.cache_path, "r")
            self._worker_handles[worker_id] = handle
        return handle

    def close(self) -> None:
        if self._main_handle is not None:
            self._main_handle.close()
            self._main_handle = None
        for handle in self._worker_handles.values():
            handle.close()
        self._worker_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # legacy (on-the-fly) extraction
    def _extract_all_patches(self, preload: bool = False):
        patches = []
        for aoi_idx, aoi_name in enumerate(self.aoi_list):
            aoi_path = self.root_dir / aoi_name
            if not aoi_path.exists():
                print(f"warning: aoi {aoi_name} not found, skipping")
                continue

            try:
                images, masks, valid_mask = load_temporal_sequence(
                    aoi_path, max_seq_len=self.max_seq_len, use_bands=self.use_bands
                )
                images = images[valid_mask]
                masks = masks[valid_mask]

                if len(images) == 0:
                    continue

                aoi_patches = extract_patches(
                    images, masks, patch_size=self.patch_size, stride=self.patch_size
                )

                for patch_imgs, patch_masks, position in aoi_patches:
                    if preload:
                        patches.append(
                            {
                                "images": patch_imgs,
                                "masks": patch_masks,
                                "aoi_name": aoi_name,
                                "position": position,
                            }
                        )
                    else:
                        patches.append(
                            {"aoi_name": aoi_name, "position": position, "images": None, "masks": None}
                        )

                if (aoi_idx + 1) % 10 == 0:
                    print(f"  processed {aoi_idx + 1}/{len(self.aoi_list)} aois")

            except Exception as exc:
                print(f"error processing {aoi_name}: {exc}")
                continue

        return patches

    # ------------------------------------------------------------------ #
    # dataset API
    def __len__(self) -> int:
        return self.num_patches

    def _sample_from_hdf5(self, idx: int):
        handle = self._get_h5_file()
        length = int(handle["lengths"][idx])
        images = np.asarray(handle["images"][idx, :length], dtype=np.float32)
        masks = np.asarray(handle["masks"][idx, 0], dtype=np.uint8)
        doy = np.asarray(handle["doy"][idx, :length], dtype=np.float32)
        aoi_name = (
            self._aoi_names_cache[idx]
            if self._aoi_names_cache is not None
            else handle["aoi_name"][idx].decode("utf-8")
        )
        position = (
            tuple(self._positions_cache[idx])
            if self._positions_cache is not None
            else tuple(np.asarray(handle["top_left"][idx], dtype=int))
        )
        sample = {"img": images, "labels": [masks[np.newaxis, :, :]], "doy": doy}
        return sample, aoi_name, position

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.use_hdf5:
            sample, aoi_name, position = self._sample_from_hdf5(idx)
        else:
            patch_meta = self.patches[idx]
            if patch_meta["images"] is None:
                aoi_path = self.root_dir / patch_meta["aoi_name"]
                images, masks, valid_mask = load_temporal_sequence(
                    aoi_path, max_seq_len=self.max_seq_len, use_bands=self.use_bands
                )
                images = images[valid_mask]
                masks = masks[valid_mask]
                top, left = patch_meta["position"]
                patch_imgs = images[:, top : top + self.patch_size, left : left + self.patch_size, :]
                patch_masks = masks[:, top : top + self.patch_size, left : left + self.patch_size]
            else:
                patch_imgs = patch_meta["images"]
                patch_masks = patch_meta["masks"]

            T = len(patch_imgs)
            sample = {
                "img": np.asarray(patch_imgs, dtype=np.float32),
                "labels": [np.asarray(patch_masks[-1:], dtype=np.uint8)],
                "doy": np.linspace(0, 365, T, dtype=np.float32),
            }
            aoi_name = patch_meta["aoi_name"]
            position = patch_meta["position"]

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, f"{aoi_name}_{position}"

        return sample

    def read(self, idx, abs: bool = False):
        if not isinstance(idx, int):
            raise ValueError("idx must be integer for spacenet7 dataset")

        if self.use_hdf5:
            sample, _, _ = self._sample_from_hdf5(idx)
            return sample

        patch_meta = self.patches[idx]
        aoi_path = self.root_dir / patch_meta["aoi_name"]
        images, masks, valid_mask = load_temporal_sequence(
            aoi_path, max_seq_len=self.max_seq_len, use_bands=self.use_bands
        )
        images = images[valid_mask]
        masks = masks[valid_mask]
        top, left = patch_meta["position"]
        patch_imgs = images[:, top : top + self.patch_size, left : left + self.patch_size, :]
        patch_masks = masks[:, top : top + self.patch_size, left : left + self.patch_size]

        T = len(patch_imgs)
        sample = {
            "img": patch_imgs,
            "labels": [patch_masks[-1:, :, :]],
            "doy": np.linspace(0, 365, T).astype(np.float32),
        }
        return sample


def my_collate(batch):
    """
    filter out samples where mask is zero everywhere.
    compatible with pastis24 collate function.
    """
    idx = [b["labels"].sum() > 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]

    if len(batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(batch)