import torch
import sys

print("=" * 80)
print("environment verification")
print("=" * 80)

# check pytorch and cuda
print(f"\npytorch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"cuda version: {torch.version.cuda}")
    print(f"cudnn version: {torch.backends.cudnn.version()}")
    print(f"gpu devices: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\ngpu {i}:")
        print(f"  name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  memory: {props.total_memory / 1e9:.2f} gb")
        print(f"  compute capability: {props.major}.{props.minor}")
    
    # test gpu computation
    print("\ntesting gpu computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"gpu computation test: successful (result shape: {z.shape})")
else:
    print("\nwarning: cuda not available. training will be slow on cpu.")
    sys.exit(1)

# check key dependencies
print("\n" + "=" * 80)
print("dependency check")
print("=" * 80)

dependencies = {
    'rasterio': None,
    'geopandas': None,
    'shapely': None,
    'einops': None,
    'mmcv': None,
    'tensorboard': None,
    'tqdm': None,
    'numpy': None,
    'pandas': None,
    'sklearn': None,
}

for dep in dependencies:
    try:
        if dep == 'sklearn':
            module = __import__('sklearn')
        else:
            module = __import__(dep)
        version = getattr(module, '__version__', 'unknown')
        print(f"{dep:15s} {version}")
        dependencies[dep] = True
    except ImportError:
        print(f"{dep:15s} not installed")
        dependencies[dep] = False

# summary
print("\n" + "=" * 80)
print("summary")
print("=" * 80)

if torch.cuda.is_available() and all(dependencies.values()):
    print("\nall checks passed. ready for implementation.")
else:
    print("\nsome checks failed:")
    if not torch.cuda.is_available():
        print("  - cuda not available")
    failed_deps = [k for k, v in dependencies.items() if not v]
    if failed_deps:
        print(f"  - missing dependencies: {', '.join(failed_deps)}")
    sys.exit(1)

print("=" * 80)