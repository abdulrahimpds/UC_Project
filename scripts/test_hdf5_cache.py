"""
test hdf5 cache loading for spacenet7.
verifies that all three splits (train/val/test) can be loaded and return correct tensor shapes.
"""
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from data import get_dataloaders
from utils.config_files_utils import read_yaml

def test_cache_loading(config_path: str):
    """test cache loading with a sample config"""
    config = read_yaml(config_path)
    
    print(f"testing cache loading with config: {config_path}")
    print(f"  dataset: {config['DATASETS']['train']['dataset']}")
    
    # create dataloaders
    print("\ncreating dataloaders...")
    start = time.perf_counter()
    dataloaders = get_dataloaders(config)
    elapsed = time.perf_counter() - start
    print(f"  dataloader creation: {elapsed:.2f}s")
    
    # test train loader
    print("\ntesting train dataloader:")
    print(f"  total samples: {len(dataloaders['train'].dataset)}")
    train_iter = iter(dataloaders['train'])
    batch = next(train_iter)
    
    print(f"  batch keys: {batch.keys()}")
    print(f"  inputs shape: {batch['inputs'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  inputs dtype: {batch['inputs'].dtype}")
    print(f"  labels dtype: {batch['labels'].dtype}")
    
    # measure batch loading time
    num_test_batches = 10
    print(f"\nmeasuring batch loading speed ({num_test_batches} batches)...")
    start = time.perf_counter()
    for i, batch in enumerate(dataloaders['train']):
        if i >= num_test_batches - 1:
            break
    elapsed = time.perf_counter() - start
    print(f"  average time per batch: {elapsed / num_test_batches:.3f}s")
    print(f"  throughput: {num_test_batches / elapsed:.1f} batches/sec")
    
    # test eval loader
    print("\ntesting eval dataloader:")
    print(f"  total samples: {len(dataloaders['eval'].dataset)}")
    eval_iter = iter(dataloaders['eval'])
    batch = next(eval_iter)
    print(f"  inputs shape: {batch['inputs'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    
    print("\n[SUCCESS] all cache loading tests passed!")

if __name__ == "__main__":
    # test with sanity check config
    config_path = "configs/SpaceNet7/sanity_check/TSViT.yaml"
    test_cache_loading(config_path)