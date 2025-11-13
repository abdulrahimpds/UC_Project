"""
test spacenet7 dataloader end-to-end
verifies data pipeline before training
"""
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from data import get_dataloaders
from utils.config_files_utils import read_yaml
import numpy as np


def test_spacenet7_dataloader():
    """test spacenet7 data loading pipeline"""
    
    print("=" * 80)
    print("testing spacenet7 dataloader")
    print("=" * 80)
    
    # load config
    config_path = "configs/SpaceNet7/TSViT.yaml"
    config = read_yaml(config_path)
    print(f"\nloaded config from {config_path}")
    print(f"  model: {config['MODEL']['architecture']}")
    print(f"  dataset: {config['DATASETS']['train']['dataset']}")
    print(f"  batch_size: {config['DATASETS']['train']['batch_size']}")
    
    # create dataloaders
    print("\ncreating dataloaders...")
    try:
        dataloaders = get_dataloaders(config)
        print("[OK] dataloaders created successfully")
    except Exception as e:
        print(f"[ERROR] error creating dataloaders: {e}")
        raise
    
    # test train loader
    print("\n" + "-" * 80)
    print("testing train dataloader")
    print("-" * 80)
    
    train_loader = dataloaders['train']
    print(f"train dataset size: {len(train_loader.dataset)}")
    
    try:
        for i, batch in enumerate(train_loader):
            print(f"\nbatch {i+1}:")
            print(f"  inputs shape: {batch['inputs'].shape}")  # [B, T, H, W, C]
            print(f"  labels shape: {batch['labels'].shape}")  # [B, H, W, 1]
            print(f"  seq_lengths: {batch['seq_lengths']}")
            
            # check data ranges
            print(f"\n  data statistics:")
            print(f"    inputs - min: {batch['inputs'].min():.3f}, max: {batch['inputs'].max():.3f}")
            print(f"    inputs - mean: {batch['inputs'].mean():.3f}, std: {batch['inputs'].std():.3f}")
            print(f"    labels - unique values: {batch['labels'].unique().tolist()}")
            
            # check for unk_masks
            if 'unk_masks' in batch:
                print(f"    unk_masks shape: {batch['unk_masks'].shape}")
                print(f"    valid pixels: {batch['unk_masks'].sum().item()} / {batch['unk_masks'].numel()}")
            
            if i >= 2:  # test first 3 batches
                break
        
        print("\n[OK] train dataloader working correctly")
        
    except Exception as e:
        print(f"\n[ERROR] error in train dataloader: {e}")
        raise
    
    # test eval loader
    print("\n" + "-" * 80)
    print("testing eval dataloader")
    print("-" * 80)
    
    eval_loader = dataloaders['eval']
    print(f"eval dataset size: {len(eval_loader.dataset)}")
    
    try:
        for i, batch in enumerate(eval_loader):
            print(f"\nbatch {i+1}:")
            print(f"  inputs shape: {batch['inputs'].shape}")
            print(f"  labels shape: {batch['labels'].shape}")
            
            if i >= 1:  # test first 2 batches
                break
        
        print("\n[OK] eval dataloader working correctly")
        
    except Exception as e:
        print(f"\n[ERROR] error in eval dataloader: {e}")
        raise
    
    # additional checks
    print("\n" + "-" * 80)
    print("data pipeline validation")
    print("-" * 80)
    
    # check dimensions match config
    expected_T = config['MODEL']['max_seq_len']
    expected_C = config['MODEL']['num_channels']
    expected_H = config['MODEL']['img_res']
    expected_W = config['MODEL']['img_res']
    
    print(f"\nexpected dimensions:")
    print(f"  temporal: {expected_T}")
    print(f"  channels: {expected_C}")
    print(f"  spatial: {expected_H}x{expected_W}")
    
    # get one batch to check
    batch = next(iter(train_loader))
    B, T, H, W, C = batch['inputs'].shape
    
    print(f"\nactual dimensions:")
    print(f"  batch: {B}")
    print(f"  temporal: {T}")
    print(f"  spatial: {H}x{W}")
    print(f"  channels: {C}")
    
    assert T == expected_T, f"temporal dim mismatch: {T} != {expected_T}"
    assert H == expected_H and W == expected_W, f"spatial dim mismatch: {H}x{W} != {expected_H}x{expected_W}"
    assert C == expected_C, f"channel dim mismatch: {C} != {expected_C}"
    
    print("\n[OK] all dimensions match config")
    
    # check label values
    unique_labels = batch['labels'].unique()
    print(f"\nunique label values: {unique_labels.tolist()}")
    assert all(l >= 0 and l <= 1 for l in unique_labels), "labels must be binary (0 or 1)"
    print("[OK] labels are binary")
    
    # check for nans
    assert not torch.isnan(batch['inputs']).any(), "found nans in inputs"
    assert not torch.isnan(batch['labels']).any(), "found nans in labels"
    print("[OK] no nans in data")
    
    print("\n" + "=" * 80)
    print("[OK] all tests passed! data pipeline is ready")
    print("=" * 80)


if __name__ == "__main__":
    test_spacenet7_dataloader()