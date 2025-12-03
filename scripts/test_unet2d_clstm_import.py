"""
Test if UNet2D-CLSTM can be imported and initialized without torchfcn dependency issues
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device

print("="*60)
print("Testing UNet2D-CLSTM Import (torchfcn fix validation)")
print("="*60)

try:
    print("\n[1/4] Loading config...")
    config = read_yaml('configs/SpaceNet7/UNet2D_CLSTM_run1.yaml')
    print("[PASS] Config loaded successfully")
    
    print("\n[2/4] Getting device...")
    device = get_device([0], allow_cpu=False)
    print(f"[PASS] Device: {device}")
    
    print("\n[3/4] Importing model (this is where torchfcn error would occur)...")
    from models import get_model
    print("[PASS] Model import successful (no torchfcn at module level)")
    
    print("\n[4/4] Instantiating UNet2D-CLSTM model...")
    model = get_model(config, device)
    print(f"[PASS] Model created: {type(model).__name__}")
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    
    print("\n[5/5] Testing forward pass with dummy input...")
    dummy_input = torch.randn(2, 24, 64, 64, 4).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"[PASS] Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED - UNet2D-CLSTM works without torchfcn!")
    print("="*60)
    sys.exit(0)
    
except ImportError as e:
    print(f"\n[FAIL] IMPORT ERROR: {e}")
    print("\n[WARNING] torchfcn fix did NOT work - dependency chain issue detected")
    sys.exit(1)
    
except Exception as e:
    print(f"\n[FAIL] ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)