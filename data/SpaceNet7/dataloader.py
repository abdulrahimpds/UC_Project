"""
spacenet7 dataloader for urban building segmentation
follows pastis24 pattern for compatibility with existing training pipeline
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data
from pathlib import Path
import pickle
from .preprocessing import load_temporal_sequence, extract_patches


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, 
                   shuffle=True, return_paths=False, my_collate=None):
    """
    create dataloader for spacenet7 dataset
    
    args:
        paths_file: path to csv file listing aoi identifiers
        root_dir: base directory containing spacenet7 data
        transform: optional transforms to apply
        batch_size: batch size
        num_workers: number of worker processes
        shuffle: whether to shuffle data
        return_paths: whether to return file paths with samples
        my_collate: optional custom collate function
    
    returns:
        pytorch dataloader
    """
    dataset = SpaceNet7Dataset(
        csv_file=paths_file, 
        root_dir=root_dir, 
        transform=transform, 
        return_paths=return_paths
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=my_collate
    )
    return dataloader


class SpaceNet7Dataset(Dataset):
    """
    spacenet7 dataset for temporal urban building segmentation
    
    loads multi-temporal satellite images and building footprint labels
    returns patches following pastis24 format for compatibility
    """
    
    def __init__(self, csv_file, root_dir, transform=None, return_paths=False,
                 patch_size=64, max_seq_len=24, use_bands=[0, 1, 2], 
                 cache_dir=None, preload=False):
        """
        args:
            csv_file: path to csv file with aoi identifiers (one per line, no header)
            root_dir: directory containing spacenet7 train/ subdirectory
            transform: optional transform pipeline
            return_paths: if true, return file paths with samples
            patch_size: size of spatial patches to extract
            max_seq_len: maximum temporal sequence length
            use_bands: which image bands to use (0-indexed), default [0,1,2] for rgb
            cache_dir: optional directory to save/load preprocessed patches
            preload: if true, preload all data into memory
        """
        # load aoi list
        if type(csv_file) == str:
            self.aoi_list = pd.read_csv(csv_file, header=None)[0].tolist()
        elif type(csv_file) in [list, tuple]:
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
        self.use_bands = use_bands
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # extract patches from all aois
        print(f"loading spacenet7 dataset from {len(self.aoi_list)} aois...")
        self.patches = self._extract_all_patches(preload=preload)
        print(f"total patches: {len(self.patches)}")
    
    def _extract_all_patches(self, preload=False):
        """
        extract patches from all aois
        
        args:
            preload: if true, preload actual data; otherwise store metadata only
        
        returns:
            list of patch metadata or preloaded data
        """
        patches = []
        
        for aoi_idx, aoi_name in enumerate(self.aoi_list):
            aoi_path = self.root_dir / aoi_name
            
            if not aoi_path.exists():
                print(f"warning: aoi {aoi_name} not found, skipping")
                continue
            
            # check cache
            if self.cache_dir:
                cache_file = self.cache_dir / f"{aoi_name}_patches.pkl"
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        aoi_patches = pickle.load(f)
                        patches.extend(aoi_patches)
                        continue
            
            try:
                # load temporal sequence
                images, masks, valid_mask = load_temporal_sequence(
                    aoi_path, 
                    max_seq_len=self.max_seq_len,
                    use_bands=self.use_bands
                )
                
                # only use valid timesteps
                images = images[valid_mask]
                masks = masks[valid_mask]
                
                if len(images) == 0:
                    continue
                
                # extract patches
                aoi_patches = extract_patches(
                    images, 
                    masks, 
                    patch_size=self.patch_size,
                    stride=self.patch_size  # non-overlapping patches
                )
                
                # store patch metadata
                for patch_imgs, patch_masks, position in aoi_patches:
                    if preload:
                        # store actual data
                        patches.append({
                            'images': patch_imgs,
                            'masks': patch_masks,
                            'aoi_name': aoi_name,
                            'position': position
                        })
                    else:
                        # store metadata only
                        patches.append({
                            'aoi_name': aoi_name,
                            'position': position,
                            'images': None,
                            'masks': None
                        })
                
                # save to cache if requested
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = self.cache_dir / f"{aoi_name}_patches.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(aoi_patches, f)
                
                if (aoi_idx + 1) % 10 == 0:
                    print(f"  processed {aoi_idx + 1}/{len(self.aoi_list)} aois")
                    
            except Exception as e:
                print(f"error processing {aoi_name}: {e}")
                continue
        
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        """
        get one patch sample
        
        returns:
            sample dict with keys:
                - 'img': temporal image sequence [T, H, W, C]
                - 'labels': segmentation mask [1, H, W]
                - 'doy': day of year for each timestep [T]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        patch_meta = self.patches[idx]
        
        # load data if not preloaded
        if patch_meta['images'] is None:
            aoi_path = self.root_dir / patch_meta['aoi_name']
            images, masks, valid_mask = load_temporal_sequence(
                aoi_path,
                max_seq_len=self.max_seq_len,
                use_bands=self.use_bands
            )
            images = images[valid_mask]
            masks = masks[valid_mask]
            
            # extract specific patch
            top, left = patch_meta['position']
            patch_imgs = images[:, top:top+self.patch_size, left:left+self.patch_size, :]
            patch_masks = masks[:, top:top+self.patch_size, left:left+self.patch_size]
        else:
            patch_imgs = patch_meta['images']
            patch_masks = patch_meta['masks']
        
        # prepare sample in pastis24 format
        T = len(patch_imgs)
        
        # for temporal models, we use all timesteps
        # take last frame's mask as target (or majority vote could be used)
        sample = {
            'img': patch_imgs,  # [T, H, W, C]
            'labels': [patch_masks[-1:, :, :]],  # [1, H, W] - last timestep
            'doy': np.linspace(0, 365, T).astype(np.float32)  # placeholder temporal encoding
        }
        
        # apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        if self.return_paths:
            return sample, f"{patch_meta['aoi_name']}_{patch_meta['position']}"
        
        return sample
    
    def read(self, idx, abs=False):
        """
        read single dataset sample without transforms
        for compatibility with pastis24 interface
        """
        if type(idx) == int:
            patch_meta = self.patches[idx]
        else:
            raise ValueError("idx must be integer for spacenet7 dataset")
        
        # load data
        aoi_path = self.root_dir / patch_meta['aoi_name']
        images, masks, valid_mask = load_temporal_sequence(
            aoi_path,
            max_seq_len=self.max_seq_len,
            use_bands=self.use_bands
        )
        images = images[valid_mask]
        masks = masks[valid_mask]
        
        # extract specific patch
        top, left = patch_meta['position']
        patch_imgs = images[:, top:top+self.patch_size, left:left+self.patch_size, :]
        patch_masks = masks[:, top:top+self.patch_size, left:left+self.patch_size]
        
        T = len(patch_imgs)
        sample = {
            'img': patch_imgs,
            'labels': [patch_masks[-1:, :, :]],
            'doy': np.linspace(0, 365, T).astype(np.float32)
        }
        
        return sample


def my_collate(batch):
    """
    filter out samples where mask is zero everywhere
    compatible with pastis24 collate function
    """
    # filter out empty patches
    idx = [b['labels'].sum() > 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    
    if len(batch) == 0:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)