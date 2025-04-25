# -*- coding: utf-8 -*-
"""
Test script for SFDFuse model.
This script evaluates the model's performance on different datasets and saves fusion results.
"""

import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
from processing import get_dataloader
import warnings
import logging
import cv2
from net import UnifiedFusionNetwork
import glob

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ckpt_path = r"models/best_model.pth"

def save_fused_image(fused_img, save_path, index):
    """
    Save the fused image to specified path.
    
    Args:
        fused_img: The fused image array
        save_path: Directory to save the image
        index: Image index for naming
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_name = os.path.join(save_path, f'fused_{index:04d}.png')
    
    # Ensure pixel values are in range [0, 255]
    fused_img = np.clip(fused_img, 0, 255).astype(np.uint8)
    
    # Save the image
    success = cv2.imwrite(save_name, fused_img)

def process_and_evaluate(dataset_name, unified_network, device, save_path=None):
    """
    Process and evaluate the model on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to evaluate
        unified_network: The fusion network model
        device: Computing device (CPU/GPU)
        save_path: Path to save fusion results
        
    Returns:
        numpy.ndarray: Array of evaluation metrics
    """
    print("\n"*2+"="*80)
    print(f"Testing on {dataset_name} dataset:")
    
    ir_dir = os.path.join('test_img', dataset_name, 'ir')
    vis_dir = os.path.join('test_img', dataset_name, 'vi')
    
    if not os.path.exists(ir_dir):
        raise ValueError(f"IR directory not found: {ir_dir}")
    if not os.path.exists(vis_dir):
        raise ValueError(f"VIS directory not found: {vis_dir}")
        
    ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.*")))
    vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    print(f"Found {len(ir_files)} IR images and {len(vis_files)} VIS images")
    print(f"Will save results to: {save_path}")
    
    # Reset global preprocessor
    import processing
    processing._preprocessor_instance = None
    
    test_loader = get_dataloader(
        ir_dir=ir_dir,
        vis_dir=vis_dir,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    metric_result = np.zeros((6))
    unified_network.eval()
    
    with torch.no_grad():
        for i, (vis_img, ir_img, mask, sizes) in enumerate(test_loader):
            try:
                vis_img = vis_img.to(device)
                ir_img = ir_img.to(device)
                
                # Forward pass
                fused_output, freq_info = unified_network(vis_img, ir_img)
                
                # Convert to CPU and get numpy arrays
                fused_output = fused_output.cpu().numpy()[0, 0]
                ir_eval = ir_img.cpu().numpy()[0, 0]
                vis_eval = vis_img.cpu().numpy()[0, 0]
                
                # Scale values from [0,1] to [0,255] for evaluation and saving
                fused_output = (fused_output * 255).astype(np.float32)
                ir_eval = (ir_eval * 255).astype(np.float32)
                vis_eval = (vis_eval * 255).astype(np.float32)
                
                # Save fusion results
                if save_path:
                    try:
                        save_fused_image(fused_output, save_path, i)
                    except Exception as e:
                        print(f"Error saving image {i}: {str(e)}")
                
                # Calculate evaluation metrics
                current_metrics = np.array([
                    Evaluator.EN(fused_output),      # Entropy
                    Evaluator.SD(fused_output),      # Standard Deviation
                    Evaluator.SF(fused_output),      # Spatial Frequency
                    Evaluator.MI(fused_output, ir_eval, vis_eval),    # Mutual Information
                    Evaluator.VIFF(fused_output, ir_eval, vis_eval),  # Visual Information Fidelity
                    Evaluator.Qabf(fused_output, ir_eval, vis_eval)   # Quality Assessment
                ])
                
                if not np.any(np.isnan(current_metrics)):
                    metric_result += current_metrics
                else:
                    print(f"Warning: NaN metrics detected in image {i}")
                    
            except Exception as e:
                print(f"Error processing image {i} in {dataset_name}: {str(e)}")
                continue
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} images")
        
        metric_result /= len(test_loader)
        
        results = {
            'EN': metric_result[0],    # Entropy
            'SD': metric_result[1],    # Standard Deviation
            'SF': metric_result[2],    # Spatial Frequency
            'MI': metric_result[3],    # Mutual Information
            'VIF': metric_result[4],   # Visual Information Fidelity
            'Qabf': metric_result[5]   # Quality Assessment
        }
        
        print_test_results(dataset_name, results)
        return metric_result

def print_test_results(dataset_name, results):
    """
    Print evaluation results in a formatted table.
    
    Args:
        dataset_name: Name of the dataset
        results: Dictionary containing evaluation metrics
    """
    print("="*65)
    print(f"The test result of {dataset_name}")
    print("="*65)
    
    # Print header
    print("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        "Model",
        "EN",
        "SD",
        "SF",
        "MI",
        "VIF",
        "Qabf"
    ))
    
    # Print results
    print("{:<10} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f}".format(
        "SFDFuse",
        results['EN'],
        results['SD'],
        results['SF'],
        results['MI'],
        results['VIF'],
        results['Qabf']
    ))
    print()

def main():
    """
    Main function to run the evaluation process.
    Loads the model and evaluates it on multiple datasets.
    """
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    unified_network = nn.DataParallel(UnifiedFusionNetwork()).to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load(ckpt_path)
        unified_network.load_state_dict(checkpoint['unified_network'])
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading checkpoint:", e)
        print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")
        raise
    
    # Test on different datasets
    for dataset_name in ["TNO", "RoadScene"]:
        save_path = os.path.join("test_result", dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        process_and_evaluate(dataset_name, unified_network, device, save_path)

if __name__ == "__main__":
    main()

