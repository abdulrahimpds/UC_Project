"""
compute per-band normalization statistics for spacenet7 training set
"""
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from tqdm import tqdm


def compute_spacenet7_normalization_stats(train_csv="data/SpaceNet7/train_aois.csv",
                                          data_dir="data/SpaceNet7",
                                          use_bands=[0, 1, 2],
                                          sample_interval=10):
    """
    compute per-band mean and std from training set
    
    args:
        train_csv: path to training aois csv
        data_dir: base data directory
        use_bands: which bands to compute stats for (0-indexed)
        sample_interval: sample every nth pixel to reduce memory usage
    
    returns:
        mean, std arrays of shape (num_bands,)
    """
    # load training aois
    train_aois = pd.read_csv(train_csv, header=None)[0].tolist()
    data_path = Path(data_dir) / "train"
    
    print(f"computing normalization statistics from {len(train_aois)} training aois")
    print(f"using bands: {use_bands}")
    print(f"sampling every {sample_interval}th pixel")
    
    # accumulate pixel values for each band
    pixel_values = {b: [] for b in use_bands}
    total_pixels = 0
    
    for aoi_name in tqdm(train_aois, desc="processing aois"):
        aoi_path = data_path / aoi_name
        images_dir = aoi_path / "images"
        
        if not images_dir.exists():
            continue
        
        # sample first 5 images from each aoi
        image_files = sorted(list(images_dir.glob("*.tif")))[:5]
        
        for img_file in image_files:
            try:
                with rasterio.open(img_file) as src:
                    for band_idx in use_bands:
                        # read entire band
                        band_data = src.read(band_idx + 1)
                        
                        # sample pixels
                        sampled = band_data.flatten()[::sample_interval]
                        
                        # filter out zeros (potential no-data values)
                        sampled = sampled[sampled > 0]
                        
                        pixel_values[band_idx].extend(sampled.tolist())
                        total_pixels += len(sampled)
            except Exception as e:
                print(f"error reading {img_file}: {e}")
                continue
    
    print(f"\ntotal pixels sampled: {total_pixels}")
    
    # compute statistics
    means = []
    stds = []
    
    for band_idx in use_bands:
        values = np.array(pixel_values[band_idx], dtype=np.float32)
        band_mean = values.mean()
        band_std = values.std()
        
        means.append(band_mean)
        stds.append(band_std)
        
        print(f"\nband {band_idx}:")
        print(f"  samples: {len(values)}")
        print(f"  mean: {band_mean:.2f}")
        print(f"  std: {band_std:.2f}")
        print(f"  min: {values.min():.2f}, max: {values.max():.2f}")
    
    mean = np.array(means, dtype=np.float32)
    std = np.array(stds, dtype=np.float32)
    
    # save to file for use in transforms
    stats_file = Path(data_dir) / "normalization_stats.npz"
    np.savez(stats_file, mean=mean, std=std)
    print(f"\nsaved normalization statistics to {stats_file}")
    
    # print formatted for use in code
    print("\nformatted for data_transforms.py:")
    print(f"self.mean = np.array([{', '.join([f'[[{m:.2f}]]' for m in mean])}]).astype(np.float32)")
    print(f"self.std = np.array([{', '.join([f'[[{s:.2f}]]' for s in std])}]).astype(np.float32)")
    
    return mean, std


if __name__ == "__main__":
    mean, std = compute_spacenet7_normalization_stats()