"""
validate unet3d forward pass for the spacenet7 configuration.

this script mirrors the tsvit forward validation but targets the 3d unet baseline.
it instantiates the model with the spacenet7 config, crafts synthetic inputs in
[T, H, W, C] format including a normalized temporal channel, and asserts that the
resulting logits match the expected segmentation shape.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import torch

# ensure project root is on the python path for module resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.UNet3D.unet3d import UNet3D
from utils.config_files_utils import read_yaml


def generate_synthetic_batch(
    batch_size: int,
    timesteps: int,
    height: int,
    width: int,
    num_channels: int,
    device: torch.device,
) -> torch.Tensor:
    """create synthetic spacenet7-style inputs with spectral + doy channels."""
    if num_channels < 2:
        raise ValueError("num_channels must be at least 2 (spectral + doy).")

    spectral_channels = num_channels - 1

    spectral = torch.randn(batch_size, timesteps, height, width, spectral_channels, device=device)

    doy = torch.linspace(0.0, 1.0, steps=timesteps, device=device)
    doy = doy.view(1, timesteps, 1, 1, 1).expand(batch_size, timesteps, height, width, 1)

    return torch.cat([spectral, doy], dim=-1)


def instantiate_model(model_config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """instantiate unet3d on the requested device."""
    model = UNet3D(model_config)
    return model.to(device)


def main() -> None:
    config_unet3d = read_yaml("configs/SpaceNet7/UNet3D.yaml")
    model_config = config_unet3d["MODEL"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_unet3d = instantiate_model(model_config, device)

    batch_size = 2
    timesteps = model_config["max_seq_len"]
    height = model_config["img_res"]
    width = model_config["img_res"]
    num_channels = model_config["num_channels"]

    inputs = generate_synthetic_batch(batch_size, timesteps, height, width, num_channels, device)

    model_unet3d.eval()
    with torch.no_grad():
        outputs = model_unet3d(inputs)

    expected_shape = (batch_size, model_config["num_classes"], height, width)

    print("=== Forward Pass Validation ===")
    print(f"UNet3D output shape: {tuple(outputs.shape)}")
    print(f"Expected shape     : {expected_shape}")

    assert outputs.shape == expected_shape, "UNet3D output shape mismatch."

    params_unet3d = sum(p.numel() for p in model_unet3d.parameters() if p.requires_grad)

    print("\n=== Parameter Count ===")
    print(f"UNet3D : {params_unet3d:,d}")

    print("\nUNet3D forward pass checks passed successfully.")


if __name__ == "__main__":
    main()