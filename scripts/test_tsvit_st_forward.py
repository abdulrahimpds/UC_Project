"""
validate tsivt-st forward pass against the original tsivt implementation.

this script loads both temporal-first and spatial-first variants using the
spacenet7 configuration, generates synthetic inputs that mimic the expected
data format (rgb bands plus normalized day-of-year channel), and verifies that
the forward pass executes without runtime errors while producing the expected
output shape. it also compares the parameter counts to confirm architectural
parity between the two variants.
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

from models.TSViT.TSViTdense import TSViT
from models.TSViT.TSViTdense_ST import TSViT_ST
from utils.config_files_utils import read_yaml


def generate_synthetic_batch(
    batch_size: int,
    timesteps: int,
    height: int,
    width: int,
    num_channels: int,
    device: torch.device,
) -> torch.Tensor:
    """create synthetic input matching the expected spacenet7 time series format."""
    if num_channels < 2:
        raise ValueError("num_channels must be at least 2 (rgb + doy channel).")

    spectral_channels = num_channels - 1

    # random spectral data
    spectral = torch.randn(batch_size, timesteps, height, width, spectral_channels, device=device)

    # normalized day-of-year channel (monotonically increasing per timestep)
    doy = torch.linspace(0.0, 1.0, steps=timesteps, device=device)
    doy = doy.view(1, timesteps, 1, 1, 1).expand(batch_size, timesteps, height, width, 1)

    return torch.cat([spectral, doy], dim=-1)


def instantiate_model(model_cls, model_config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """instantiate model on the requested device."""
    model = model_cls(model_config)
    return model.to(device)


def main() -> None:
    config_tsvit = read_yaml("configs/SpaceNet7/TSViT.yaml")
    config_tsvit_st = read_yaml("configs/SpaceNet7/TSViT-ST.yaml")

    model_config_tsvit = config_tsvit["MODEL"]
    model_config_tsvit_st = config_tsvit_st["MODEL"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_tsvit = instantiate_model(TSViT, model_config_tsvit, device)
    model_tsvit_st = instantiate_model(TSViT_ST, model_config_tsvit_st, device)

    batch_size = 2
    timesteps = model_config_tsvit["max_seq_len"]
    height = model_config_tsvit["img_res"]
    width = model_config_tsvit["img_res"]
    num_channels = model_config_tsvit["num_channels"]

    inputs = generate_synthetic_batch(batch_size, timesteps, height, width, num_channels, device)

    model_tsvit.eval()
    model_tsvit_st.eval()

    with torch.no_grad():
        out_tsvit = model_tsvit(inputs.clone())
        out_tsvit_st = model_tsvit_st(inputs.clone())

    expected_shape = (batch_size, model_config_tsvit["num_classes"], height, width)

    print("=== Forward Pass Validation ===")
    print(f"TSViT output shape: {tuple(out_tsvit.shape)}")
    print(f"TSViT-ST output shape: {tuple(out_tsvit_st.shape)}")
    print(f"Expected shape: {expected_shape}")

    assert out_tsvit.shape == expected_shape, "TSViT output shape mismatch."
    assert out_tsvit_st.shape == expected_shape, "TSViT-ST output shape mismatch."

    params_tsvit = sum(p.numel() for p in model_tsvit.parameters() if p.requires_grad)
    params_tsvit_st = sum(p.numel() for p in model_tsvit_st.parameters() if p.requires_grad)

    print("\n=== Parameter Count ===")
    print(f"TSViT      : {params_tsvit:,d}")
    print(f"TSViT-ST   : {params_tsvit_st:,d}")
    print(f"Difference : {params_tsvit_st - params_tsvit:,d}")
    assert params_tsvit == params_tsvit_st, "Parameter counts should match."

    print("\nAll TSViT-ST forward pass checks passed successfully.")


if __name__ == "__main__":
    main()