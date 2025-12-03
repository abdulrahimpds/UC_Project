"""
training orchestration script for spacenet7 experiments.
supports running multiple models sequentially with automatic checkpointing and logging.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_training(config_path: str, run_name: str = None):
    """run training with the specified config"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    
    # determine run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config_path.stem}_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"starting training: {run_name}")
    print(f"config: {config_path}")
    print(f"{'='*80}\n")
    
    # build command
    cmd = [
        sys.executable,
        "train_and_eval/segmentation_training_transf.py",
        "--config",
        str(config_path),
        "--device",
        "0"
    ]
    
    # run training
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=False
        )
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"training completed: {run_name}")
        print(f"elapsed time: {elapsed}")
        print(f"{'='*80}\n")
        
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"training failed: {run_name}")
        print(f"elapsed time: {elapsed}")
        print(f"error: {e}")
        print(f"{'='*80}\n")
        
        return False, elapsed

def run_batch_training(configs: list, log_file: str = None):
    """run multiple training configs sequentially"""
    
    # setup logging
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_log_{timestamp}.json"
    
    log_path = PROJECT_ROOT / "models" / "training_logs" / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # track results
    results = {
        "start_time": datetime.now().isoformat(),
        "configs": configs,
        "runs": []
    }
    
    # run each config
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"batch training: {i}/{len(configs)}")
        print(f"{'#'*80}\n")
        
        success, elapsed = run_training(config)
        
        run_result = {
            "config": config,
            "success": success,
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_str": str(elapsed)
        }
        results["runs"].append(run_result)
        
        # save intermediate results
        results["end_time"] = datetime.now().isoformat()
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        
        if not success:
            print(f"\nwarning: training failed for {config}")
            print("continuing with next config...")
    
    # final summary
    print(f"\n{'='*80}")
    print("batch training complete!")
    print(f"{'='*80}")
    print(f"total configs: {len(configs)}")
    print(f"successful: {sum(1 for r in results['runs'] if r['success'])}")
    print(f"failed: {sum(1 for r in results['runs'] if not r['success'])}")
    print(f"log saved to: {log_path}")
    print(f"{'='*80}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="run training experiments")
    parser.add_argument(
        "--config",
        type=str,
        help="path to single config file"
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        help="paths to multiple config files for batch training"
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="run sanity check training for all models"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run full training for all models (3 runs each)"
    )
    parser.add_argument(
        "--log",
        type=str,
        help="log file name (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # single config
    if args.config:
        run_training(args.config)
    
    # batch configs
    elif args.batch:
        run_batch_training(args.batch, args.log)
    
    # sanity check
    elif args.sanity:
        configs = [
            "configs/SpaceNet7/sanity_check/TSViT.yaml",
            "configs/SpaceNet7/sanity_check/TSViT-ST.yaml",
            "configs/SpaceNet7/sanity_check/UNet3D.yaml",
            "configs/SpaceNet7/sanity_check/UNet2D_CLSTM.yaml"
        ]
        run_batch_training(configs, args.log or "sanity_check.json")
    
    # full experiments
    elif args.full:
        configs = []
        models = ["TSViT", "TSViT-ST", "UNet3D", "UNet2D_CLSTM"]
        for model in models:
            for run_idx in range(1, 4):
                configs.append(f"configs/SpaceNet7/{model}.yaml")
        run_batch_training(configs, args.log or "full_experiments.json")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()