"""
preprocessing utilities for spacenet7 dataset
handles geojson rasterization and data preparation
"""
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.ops import transform
import pyproj


def rasterize_buildings(geojson_path, reference_image_path, out_shape=None):
    """
    rasterize building polygons from geojson to match image grid
    
    args:
        geojson_path: path to buildings geojson file
        reference_image_path: path to reference tif image for spatial alignment
        out_shape: optional output shape (height, width)
    
    returns:
        binary mask array (height, width) with buildings as 1, background as 0
    """
    # load reference image for spatial properties
    with rasterio.open(reference_image_path) as src:
        transform = src.transform
        if out_shape is None:
            out_shape = (src.height, src.width)
        image_crs = src.crs
    
    # load building polygons
    gdf = gpd.read_file(geojson_path)
    
    # reproject polygons to match image crs
    if gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)
    
    # rasterize polygons
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    
    if len(shapes) == 0:
        # no buildings in this frame
        mask = np.zeros(out_shape, dtype=np.uint8)
    else:
        mask = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
    
    return mask


def load_temporal_sequence(aoi_path, max_seq_len=24, use_bands=[0, 1, 2]):
    """
    load temporal image sequence for one aoi
    
    args:
        aoi_path: path to aoi directory
        max_seq_len: maximum sequence length to load (default 24)
        use_bands: which bands to use (0-indexed), default [0,1,2] for rgb
    
    returns:
        images: array of shape (T, H, W, C)
        masks: array of shape (T, H, W)
        valid_mask: boolean array of shape (T,) indicating valid timesteps
    """
    from pathlib import Path
    
    aoi_path = Path(aoi_path)
    images_dir = aoi_path / "images"
    labels_dir = aoi_path / "labels"
    
    # get sorted image files
    image_files = sorted(list(images_dir.glob("*.tif")))[:max_seq_len]
    
    if len(image_files) == 0:
        raise ValueError(f"no images found in {images_dir}")
    
    # load first image to get dimensions
    with rasterio.open(image_files[0]) as src:
        height, width = src.height, src.width
        num_bands = len(use_bands)
    
    # initialize arrays
    T = len(image_files)
    images = np.zeros((T, height, width, num_bands), dtype=np.float32)
    masks = np.zeros((T, height, width), dtype=np.uint8)
    valid_mask = np.ones(T, dtype=bool)
    
    # load each timestep
    for t, img_file in enumerate(image_files):
        try:
            # load image
            with rasterio.open(img_file) as src:
                # read selected bands (1-indexed in rasterio)
                img_data = src.read([b + 1 for b in use_bands])
                images[t] = img_data.transpose(1, 2, 0)  # CHW -> HWC
            
            # load corresponding building labels
            label_file = labels_dir / img_file.name.replace(".tif", "_Buildings.geojson")
            if label_file.exists():
                masks[t] = rasterize_buildings(label_file, img_file, (height, width))
            else:
                valid_mask[t] = False
                
        except Exception as e:
            print(f"error loading timestep {t} from {img_file.name}: {e}")
            valid_mask[t] = False
    
    return images, masks, valid_mask


def extract_patches(images, masks, patch_size=64, stride=None, random=False, num_patches=None):
    """
    extract spatial patches from temporal sequences
    
    args:
        images: array of shape (T, H, W, C)
        masks: array of shape (T, H, W)
        patch_size: size of square patches
        stride: stride for patch extraction (default: patch_size for non-overlapping)
        random: if true, extract random patches instead of grid
        num_patches: number of random patches to extract (only if random=true)
    
    returns:
        list of (patch_images, patch_masks, position) tuples
    """
    T, H, W, C = images.shape
    
    if stride is None:
        stride = patch_size
    
    patches = []
    
    if random and num_patches is not None:
        # extract random patches
        for _ in range(num_patches):
            top = np.random.randint(0, H - patch_size + 1)
            left = np.random.randint(0, W - patch_size + 1)
            
            patch_imgs = images[:, top:top+patch_size, left:left+patch_size, :]
            patch_masks = masks[:, top:top+patch_size, left:left+patch_size]
            
            patches.append((patch_imgs, patch_masks, (top, left)))
    else:
        # extract grid patches
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                patch_imgs = images[:, top:top+patch_size, left:left+patch_size, :]
                patch_masks = masks[:, top:top+patch_size, left:left+patch_size]
                
                patches.append((patch_imgs, patch_masks, (top, left)))
    
    return patches


def compute_normalization_stats(aoi_list, data_dir, use_bands=[0, 1, 2], sample_size=10):
    """
    compute per-band mean and std for normalization
    
    args:
        aoi_list: list of aoi directory names
        data_dir: base data directory
        use_bands: which bands to compute stats for
        sample_size: number of aois to sample for stats computation
    
    returns:
        mean: array of shape (C,)
        std: array of shape (C,)
    """
    from pathlib import Path
    import random
    
    data_dir = Path(data_dir)
    
    # sample aois if needed
    if len(aoi_list) > sample_size:
        aoi_sample = random.sample(aoi_list, sample_size)
    else:
        aoi_sample = aoi_list
    
    # accumulate statistics
    pixel_values = {b: [] for b in use_bands}
    
    print(f"computing normalization statistics from {len(aoi_sample)} aois...")
    
    for aoi_name in aoi_sample:
        aoi_path = data_dir / aoi_name
        images_dir = aoi_path / "images"
        
        # sample a few images from each aoi
        image_files = sorted(list(images_dir.glob("*.tif")))[:5]
        
        for img_file in image_files:
            with rasterio.open(img_file) as src:
                for band_idx in use_bands:
                    band_data = src.read(band_idx + 1)
                    # sample pixels to reduce memory
                    sampled = band_data.flatten()[::100]
                    pixel_values[band_idx].extend(sampled.tolist())
    
    # compute mean and std
    means = []
    stds = []
    for band_idx in use_bands:
        values = np.array(pixel_values[band_idx])
        means.append(values.mean())
        stds.append(values.std())
    
    mean = np.array(means, dtype=np.float32)
    std = np.array(stds, dtype=np.float32)
    
    print(f"normalization stats:")
    print(f"  mean: {mean}")
    print(f"  std: {std}")
    
    return mean, std