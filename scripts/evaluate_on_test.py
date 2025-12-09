"""
evaluate trained models on the test set and save metrics to csv.
"""
import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint
from data import get_dataloaders, get_loss_data_input
from metrics.numpy_metrics import get_classification_metrics
from metrics.loss_functions import get_loss


def evaluate_on_test(net, testloader, loss_fn, config, device, loss_input_fn):
    """
    evaluate model on test set and return metrics.
    
    args:
        net: trained model
        testloader: test dataloader
        loss_fn: loss function
        config: configuration dictionary
        device: torch device
        loss_input_fn: function to extract labels from sample
        
    returns:
        dictionary of metrics (macro and micro averages, per-class metrics)
    """
    num_classes = config['MODEL']['num_classes']
    predicted_all = []
    labels_all = []
    losses_all = []
    
    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(testloader):
            logits = net(sample['inputs'].to(device))
            logits = logits.permute(0, 2, 3, 1)
            _, predicted = torch.max(logits.data, -1)
            
            ground_truth = loss_input_fn(sample, device)
            loss = loss_fn(logits, ground_truth)
            
            target, mask = ground_truth
            if mask is not None:
                predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
            else:
                predicted_all.append(predicted.view(-1).cpu().numpy())
                labels_all.append(target.view(-1).cpu().numpy())
            losses_all.append(loss.view(-1).cpu().detach().numpy())
            
            if (step + 1) % 50 == 0:
                print(f"  processed {step + 1}/{len(testloader)} batches")
    
    print(f"finished iterating over test set ({len(testloader)} batches)")
    print("calculating metrics...")
    
    predicted_classes = np.concatenate(predicted_all)
    target_classes = np.concatenate(labels_all)
    losses = np.concatenate(losses_all)
    
    eval_metrics = get_classification_metrics(
        predicted=predicted_classes, 
        labels=target_classes,
        n_classes=num_classes, 
        unk_masks=None
    )
    
    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']
    
    print("-" * 120)
    print(f"test metrics (micro/macro):")
    print(f"  loss: {losses.mean():.7f}")
    print(f"  iou: {micro_IOU:.4f} / {macro_IOU:.4f}")
    print(f"  accuracy: {micro_acc:.4f} / {macro_acc:.4f}")
    print(f"  precision: {micro_precision:.4f} / {macro_precision:.4f}")
    print(f"  recall: {micro_recall:.4f} / {macro_recall:.4f}")
    print(f"  f1: {micro_F1:.4f} / {macro_F1:.4f}")
    print(f"  unique pred labels: {np.unique(predicted_classes)}")
    print("-" * 120)
    
    return {
        "macro": {
            "Loss": losses.mean(), 
            "Accuracy": macro_acc, 
            "Precision": macro_precision,
            "Recall": macro_recall, 
            "F1": macro_F1, 
            "IOU": macro_IOU
        },
        "micro": {
            "Loss": losses.mean(), 
            "Accuracy": micro_acc, 
            "Precision": micro_precision,
            "Recall": micro_recall, 
            "F1": micro_F1, 
            "IOU": micro_IOU
        },
        "class": {
            "Accuracy": class_acc, 
            "Precision": class_precision,
            "Recall": class_recall,
            "F1": class_F1, 
            "IOU": class_IOU
        }
    }


def main():
    parser = argparse.ArgumentParser(description='evaluate model on spacenet7 test set')
    parser.add_argument('--config', required=True, help='path to config (.yaml) file')
    parser.add_argument('--checkpoint', required=True, help='path to model checkpoint (.pth)')
    parser.add_argument('--device', default='0', type=str, help='gpu id to use')
    parser.add_argument('--output', default='results/test_metrics.csv', 
                       help='path to save results csv')
    parser.add_argument('--model_name', required=True, help='name of model (e.g., TSViT_run1)')
    
    args = parser.parse_args()
    
    # setup device
    device_ids = [int(d) for d in args.device.split(',')]
    device = get_device(device_ids, allow_cpu=False)
    
    # load config
    print(f"\nloading config from {args.config}")
    config = read_yaml(args.config)
    config['local_device_ids'] = device_ids
    
    # ensure test set is in config
    if 'test' not in config['DATASETS']:
        print("adding test dataset config...")
        config['DATASETS']['test'] = config['DATASETS']['eval'].copy()
    
    # load data
    print("loading test dataloader...")
    dataloaders = get_dataloaders(config)
    testloader = dataloaders['test']
    print(f"test set has {len(testloader)} batches")
    
    # load model
    print(f"\nloading model from {args.checkpoint}")
    net = get_model(config, device)
    load_from_checkpoint(net, args.checkpoint, partial_restore=False)
    net.to(device)
    net.eval()
    
    # setup loss
    loss_input_fn = get_loss_data_input(config)
    loss_fn = get_loss(config, device, reduction=None)
    
    # evaluate
    print(f"\nevaluating {args.model_name} on test set...")
    metrics = evaluate_on_test(net, testloader, loss_fn, config, device, loss_input_fn)
    
    # save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # prepare results row
    results_row = {
        'model_name': args.model_name,
        'checkpoint': args.checkpoint,
        'macro_loss': metrics['macro']['Loss'],
        'macro_accuracy': metrics['macro']['Accuracy'],
        'macro_precision': metrics['macro']['Precision'],
        'macro_recall': metrics['macro']['Recall'],
        'macro_f1': metrics['macro']['F1'],
        'macro_iou': metrics['macro']['IOU'],
        'micro_loss': metrics['micro']['Loss'],
        'micro_accuracy': metrics['micro']['Accuracy'],
        'micro_precision': metrics['micro']['Precision'],
        'micro_recall': metrics['micro']['Recall'],
        'micro_f1': metrics['micro']['F1'],
        'micro_iou': metrics['micro']['IOU'],
    }
    
    # add per-class metrics
    for class_idx in range(len(metrics['class']['IOU'])):
        results_row[f'class_{class_idx}_iou'] = metrics['class']['IOU'][class_idx]
        results_row[f'class_{class_idx}_accuracy'] = metrics['class']['Accuracy'][class_idx]
        results_row[f'class_{class_idx}_precision'] = metrics['class']['Precision'][class_idx]
        results_row[f'class_{class_idx}_recall'] = metrics['class']['Recall'][class_idx]
        results_row[f'class_{class_idx}_f1'] = metrics['class']['F1'][class_idx]
    
    # save to csv (append if exists)
    df = pd.DataFrame([results_row])
    if output_path.exists():
        df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"\nappended results to {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"\ncreated results file at {output_path}")
    
    print(f"\nevaluation complete for {args.model_name}")
    print(f"  macro iou: {metrics['macro']['IOU']:.4f}")
    print(f"  micro iou: {metrics['micro']['IOU']:.4f}")


if __name__ == "__main__":
    main()