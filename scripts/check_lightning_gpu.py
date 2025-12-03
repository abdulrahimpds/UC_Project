"""
diagnostic script to check lightning.ai gpu setup.
"""
import sys
import torch
import subprocess

print("=" * 80)
print("lightning.ai gpu diagnostic")
print("=" * 80)

print(f"\ncurrent python: {sys.executable}")
print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"cuda version: {torch.version.cuda}")
    print(f"cudnn version: {torch.backends.cudnn.version()}")
    print(f"device count: {torch.cuda.device_count()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
else:
    print("\n[ERROR] CUDA not available!")
    print("\npossible causes:")
    print("1. lightning.ai studio created without GPU allocation")
    print("2. wrong conda environment active")
    print("3. pytorch-cpu installed instead of pytorch-cuda")
    
    print("\nsolutions:")
    print("1. create new studio with GPU (T4, L4, etc.)")
    print("2. activate correct conda env: conda activate cloudspace")
    print("3. reinstall pytorch with cuda:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 80)
print("checking subprocess python (used by run_training.py):")
print("=" * 80)

result = subprocess.run(
    [sys.executable, "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}')"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"[ERROR] {result.stderr}")

print("=" * 80)