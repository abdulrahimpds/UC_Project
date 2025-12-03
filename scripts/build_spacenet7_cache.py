"""
build a persistent hdf5 cache for spacenet7 patch tensors.

example:
    python scripts/build_spacenet7_cache.py \
        --paths-csv data/SpaceNet7/paths/train_paths.csv \
        --root-dir data/SpaceNet7 \
        --cache-path data/SpaceNet7/cache/train.h5 \
        --split train \
        --max-seq-len 24 \
        --patch-size 64 \
        --use-bands 0 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import h5py  # type: ignore
except ImportError as exc:
    raise ImportError("h5py is required to build the SpaceNet7 cache. Install with `pip install h5py`.") from exc

# ensure project root on path to reuse dataset implementation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.SpaceNet7.dataloader import SpaceNet7Dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="materialize SpaceNet7 patches into a single HDF5 file")
    parser.add_argument("--paths-csv", required=True, help="CSV file (or directory) listing AOI identifiers")
    parser.add_argument("--root-dir", required=True, help="Root directory containing SpaceNet7 data (with train/)")
    parser.add_argument("--cache-path", required=True, help="Output HDF5 file path")
    parser.add_argument("--split", default="train", help="Data split name for metadata (train/val/test)")
    parser.add_argument("--max-seq-len", type=int, default=24, help="Maximum temporal length (frames)")
    parser.add_argument("--patch-size", type=int, default=64, help="Spatial patch size (pixels)")
    parser.add_argument(
        "--use-bands",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Spectral bands to include (0-indexed)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache file")
    parser.add_argument("--no-compression", action="store_true", help="Disable gzip compression for faster builds")
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N patches")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_datasets(
    handle: h5py.File,
    num_patches: int,
    max_seq_len: int,
    patch_size: int,
    num_channels: int,
    no_compression: bool = False,
) -> None:
    compression_kwargs = {} if no_compression else {"compression": "gzip", "compression_opts": 4}
    
    chunk = (1, max_seq_len, patch_size, patch_size, num_channels)
    handle.create_dataset(
        "images",
        shape=(num_patches, max_seq_len, patch_size, patch_size, num_channels),
        maxshape=(num_patches, max_seq_len, patch_size, patch_size, num_channels),
        dtype="float32",
        chunks=chunk,
        **compression_kwargs,
    )
    handle.create_dataset(
        "masks",
        shape=(num_patches, 1, patch_size, patch_size),
        maxshape=(num_patches, 1, patch_size, patch_size),
        dtype="uint8",
        chunks=(1, 1, patch_size, patch_size),
        **compression_kwargs,
    )
    handle.create_dataset(
        "doy",
        shape=(num_patches, max_seq_len),
        maxshape=(num_patches, max_seq_len),
        dtype="float32",
        chunks=(1, max_seq_len),
        **compression_kwargs,
    )
    handle.create_dataset(
        "lengths",
        shape=(num_patches,),
        dtype="int16",
        chunks=True,
        **compression_kwargs,
    )
    string_dtype = h5py.string_dtype(encoding="utf-8", length=64)
    handle.create_dataset("aoi_name", shape=(num_patches,), dtype=string_dtype)
    handle.create_dataset("top_left", shape=(num_patches, 2), dtype="int16")


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_path)
    ensure_parent(cache_path)

    if cache_path.exists() and not args.overwrite:
        raise FileExistsError(f"{cache_path} already exists. Pass --overwrite to replace it.")

    print("preparing SpaceNet7 dataset...")
    dataset = SpaceNet7Dataset(
        csv_file=args.paths_csv,
        root_dir=args.root_dir,
        transform=None,
        return_paths=False,
        patch_size=args.patch_size,
        max_seq_len=args.max_seq_len,
        use_bands=list(args.use_bands),
        cache_dir=None,
        preload=False,
        cache_path=None,
        use_hdf5=False,
    )

    num_patches = len(dataset)
    num_channels = len(args.use_bands)

    print(f"total patches: {num_patches}")
    if num_patches == 0:
        raise RuntimeError("No patches found; check input CSV and root directory.")

    compression_mode = "disabled" if args.no_compression else "gzip level 4"
    print(f"writing cache to {cache_path} (compression: {compression_mode})...")
    with h5py.File(cache_path, "w") as h5:
        create_datasets(h5, num_patches, args.max_seq_len, args.patch_size, num_channels, args.no_compression)
        h5.attrs["split"] = args.split
        h5.attrs["max_seq_len"] = args.max_seq_len
        h5.attrs["patch_size"] = args.patch_size
        h5.attrs["bands"] = np.array(args.use_bands, dtype="int16")
        h5.attrs["created_from"] = str(args.paths_csv)

        start_time = time.perf_counter()
        for idx in range(num_patches):
            patch_meta = dataset.patches[idx]
            sample = dataset.read(idx)

            images = np.asarray(sample["img"], dtype=np.float32)
            masks = np.asarray(sample["labels"][0], dtype=np.uint8)  # [H, W]
            doy = np.asarray(sample["doy"], dtype=np.float32)
            length = images.shape[0]

            if images.shape[0] > args.max_seq_len:
                raise ValueError(
                    f"sample {idx} has temporal length {images.shape[0]} exceeding max_seq_len={args.max_seq_len}"
                )

            h5["images"][idx, :length] = images
            h5["masks"][idx, 0] = masks
            h5["doy"][idx, :length] = doy
            h5["lengths"][idx] = length
            h5["aoi_name"][idx] = patch_meta["aoi_name"]
            h5["top_left"][idx] = patch_meta["position"]

            if (idx + 1) % args.progress_every == 0 or (idx + 1) == num_patches:
                elapsed = time.perf_counter() - start_time
                print(f"[{idx + 1}/{num_patches}] cached in {elapsed:.1f}s")

    print("cache creation complete.")
    print(f"result: {cache_path}")


if __name__ == "__main__":
    main()