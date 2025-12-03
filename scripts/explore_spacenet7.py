"""
spacenet 7 data exploration script
verifies image and label characteristics for dataloader implementation
"""
import os
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from collections import defaultdict

def explore_spacenet7():
    """explore spacenet 7 dataset structure and characteristics"""
    
    # paths
    data_dir = Path("data/SpaceNet7/train")
    aois = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print("=" * 80)
    print("spacenet 7 dataset exploration")
    print("=" * 80)
    
    # basic statistics
    print(f"\ntotal aois: {len(aois)}")
    print(f"first aoi: {aois[0].name}")
    print(f"last aoi: {aois[-1].name}")
    
    # sample aoi
    sample_aoi = aois[0]
    print(f"\nexploring sample aoi: {sample_aoi.name}")
    
    # list subdirectories
    subdirs = [d.name for d in sample_aoi.iterdir() if d.is_dir()]
    print(f"subdirectories: {', '.join(subdirs)}")
    
    # image analysis
    print("\n" + "-" * 80)
    print("image analysis")
    print("-" * 80)
    
    images_dir = sample_aoi / "images"
    images = sorted(list(images_dir.glob("*.tif")))
    print(f"total images: {len(images)}")
    
    if images:
        # temporal sequence
        dates = []
        for img in images:
            parts = img.stem.split("_")
            year_month = f"{parts[2]}_{parts[3]}"
            dates.append(year_month)
        
        print(f"temporal range: {dates[0]} to {dates[-1]}")
        print(f"temporal spacing: monthly")
        
        # load first image
        sample_img = images[0]
        print(f"\nanalyzing: {sample_img.name}")
        
        with rasterio.open(sample_img) as src:
            print(f"  dimensions: {src.height} x {src.width}")
            print(f"  bands: {src.count}")
            print(f"  data type: {src.dtypes[0]}")
            print(f"  crs: {src.crs}")
            print(f"  resolution: {src.res}")
            print(f"  nodata value: {src.nodata}")
            
            # read sample region
            window = rasterio.windows.Window(0, 0, 100, 100)
            sample_data = src.read(window=window)
            
            print(f"\nsample data statistics (100x100 region):")
            for i in range(src.count):
                band_data = sample_data[i]
                print(f"  band {i+1}:")
                print(f"    min: {band_data.min()}, max: {band_data.max()}")
                print(f"    mean: {band_data.mean():.2f}, std: {band_data.std():.2f}")
    
    # label analysis
    print("\n" + "-" * 80)
    print("label analysis")
    print("-" * 80)
    
    labels_dir = sample_aoi / "labels"
    labels = sorted(list(labels_dir.glob("*_Buildings.geojson")))
    print(f"total label files: {len(labels)}")
    
    if labels:
        sample_label = labels[0]
        print(f"\nanalyzing: {sample_label.name}")
        
        gdf = gpd.read_file(sample_label)
        print(f"  building polygons: {len(gdf)}")
        print(f"  columns: {', '.join(gdf.columns.tolist())}")
        print(f"  geometry types: {gdf.geometry.type.unique().tolist()}")
        print(f"  crs: {gdf.crs}")
        
        if len(gdf) > 0:
            print(f"\nsample building attributes:")
            print(gdf.head())
    
    # udm analysis
    udm_files = sorted(list(labels_dir.glob("*_UDM.geojson")))
    print(f"\nudm (usable data mask) files: {len(udm_files)}")
    
    # compute statistics across all aois
    print("\n" + "-" * 80)
    print("dataset-wide statistics")
    print("-" * 80)
    
    print("\ncomputing statistics across all aois...")
    aoi_image_counts = []
    total_buildings = 0
    
    for aoi in aois[:10]:  # sample first 10 for speed
        images_dir = aoi / "images"
        images = list(images_dir.glob("*.tif"))
        aoi_image_counts.append(len(images))
        
        labels_dir = aoi / "labels"
        building_labels = list(labels_dir.glob("*_Buildings.geojson"))
        for label_file in building_labels:
            gdf = gpd.read_file(label_file)
            total_buildings += len(gdf)
    
    print(f"temporal sequences per aoi (sample of 10):")
    print(f"  min: {min(aoi_image_counts)}, max: {max(aoi_image_counts)}")
    print(f"  mean: {np.mean(aoi_image_counts):.1f}")
    
    print(f"\ntotal buildings in sample: {total_buildings}")
    print(f"average buildings per timestep: {total_buildings / (len(aoi_image_counts) * np.mean(aoi_image_counts)):.1f}")
    
    # recommendations
    print("\n" + "=" * 80)
    print("recommendations for dataloader implementation")
    print("=" * 80)
    
    print("\n1. temporal dimension:")
    print(f"   - use T=24 (drop last month for clean 24-month sequence)")
    print(f"   - monthly temporal spacing")
    
    print("\n2. spatial dimension:")
    with rasterio.open(images[0]) as src:
        h, w = src.height, src.width
        print(f"   - image size: {h}x{w}")
        print(f"   - recommend patch size: 64x64 or 128x128")
        print(f"   - patches per image (64x64): {(h//64) * (w//64)}")
    
    print("\n3. label rasterization:")
    print(f"   - convert geojson polygons to binary masks")
    print(f"   - align with image grid using rasterio.features.rasterize()")
    
    print("\n4. normalization:")
    print(f"   - compute per-band mean/std from training set")
    print(f"   - apply z-score normalization: (x - mean) / std")
    
    print("\n5. data splits:")
    print(f"   - split by aoi to prevent spatial leakage")
    print(f"   - recommended: 70% train / 15% val / 15% test")
    print(f"   - {len(aois)} total aois -> 42 train / 9 val / 9 test")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    explore_spacenet7()