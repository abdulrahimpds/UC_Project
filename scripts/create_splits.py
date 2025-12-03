"""
create train/val/test splits for spacenet7 dataset
splits by aoi to prevent spatial leakage
"""
import pandas as pd
from pathlib import Path
import random

def create_spacenet7_splits(data_dir="data/SpaceNet7", train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    create train/val/test splits from spacenet7 aois
    
    args:
        data_dir: path to spacenet7 data directory
        train_ratio: fraction of data for training
        val_ratio: fraction of data for validation (test gets remainder)
        seed: random seed for reproducibility
    """
    data_path = Path(data_dir) / "train"
    
    # get all aoi directories
    aois = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"found {len(aois)} aois")
    
    # shuffle with fixed seed
    random.seed(seed)
    random.shuffle(aois)
    
    # calculate split sizes
    n_total = len(aois)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # create splits
    train_aois = aois[:n_train]
    val_aois = aois[n_train:n_train+n_val]
    test_aois = aois[n_train+n_val:]
    
    print(f"\nsplit sizes:")
    print(f"  train: {len(train_aois)} ({len(train_aois)/n_total*100:.1f}%)")
    print(f"  val:   {len(val_aois)} ({len(val_aois)/n_total*100:.1f}%)")
    print(f"  test:  {len(test_aois)} ({len(test_aois)/n_total*100:.1f}%)")
    
    # save to csv files (no header, one aoi per line)
    output_dir = Path(data_dir)
    
    pd.DataFrame(train_aois).to_csv(output_dir / "train_aois.csv", index=False, header=False)
    pd.DataFrame(val_aois).to_csv(output_dir / "val_aois.csv", index=False, header=False)
    pd.DataFrame(test_aois).to_csv(output_dir / "test_aois.csv", index=False, header=False)
    
    print(f"\nsaved split files to {output_dir}/")
    print(f"  train_aois.csv")
    print(f"  val_aois.csv")
    print(f"  test_aois.csv")
    
    # display sample aois from each split
    print(f"\nsample train aois (first 5): {train_aois[:5]}")
    print(f"sample val aois:   {val_aois[:5]}")
    print(f"sample test aois:  {test_aois[:5]}")
    
    return train_aois, val_aois, test_aois


if __name__ == "__main__":
    create_spacenet7_splits()