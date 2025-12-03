"""
validate unet2d-clstm forward pass for the spacenet7 configuration.

this script instantiates the unet2d-clstm baseline (fcn_crnn / clstm segmenter) using
the spacenet7 configuration, synthesises spacenet7-style inputs in [batch, time,
height, width, channels] format (rgb bands plus a normalised day-of-year channel),
and checks that the forward pass produces logits with the expected segmentation
shape. it also reports the number of trainable parameters for bookkeeping.
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

from models.CropTypeMapping.models import FCN_CRNN  # noqa: E402
from utils.config_files_utils import read_yaml  # noqa: E402


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


def instantiate_model(model_config: Dict[str, Any]) -> torch.nn.Module:
    """
    instantiate the unet2d-clstm baseline.

    note: the underlying implementation places modules on cuda() internally,
    so this helper assumes a cuda-capable device is available.
    """
    model = FCN_CRNN(model_config)
    return model.cuda()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "unet2d-clstm forward validation requires a cuda-capable device because "
            "the baseline initialises its submodules directly on the gpu."
        )

    config = read_yaml("configs/SpaceNet7/UNet2D_CLSTM.yaml")
    model_config = config["MODEL"]

    device = torch.device("cuda")

    model_unet2d_clstm = instantiate_model(model_config)
    model_unet2d_clstm.eval()

    batch_size = 2
    timesteps = model_config["max_seq_len"]
    height = model_config["img_res"]
    width = model_config["img_res"]
    num_channels = model_config["num_channels"]

    inputs = generate_synthetic_batch(batch_size, timesteps, height, width, num_channels, device)

    with torch.no_grad():
        outputs = model_unet2d_clstm(inputs)

    expected_shape = (batch_size, model_config["num_classes"], height, width)

    print("=== Forward Pass Validation ===")
    print(f"UNet2D-CLSTM output shape: {tuple(outputs.shape)}")
    print(f"Expected shape           : {expected_shape}")

    assert outputs.shape == expected_shape, "UNet2D-CLSTM output shape mismatch."

    params_unet2d_clstm = sum(p.numel() for p in model_unet2d_clstm.parameters() if p.requires_grad)

    print("\n=== Parameter Count ===")
    print(f"UNet2D-CLSTM : {params_unet2d_clstm:,d}")

    print("\nUNet2D-CLSTM forward pass checks passed successfully.")


if __name__ == "__main__":
    main()