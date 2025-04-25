# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
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

def save_fused_image(fused_img, save_path, original_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    base_name = os.path.splitext(os.path.basename(original_name))[0]
    save_name = os.path.join(save_path, f'{base_name}.png')
    
    fused_img = np.clip(fused_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite(save_name, fused_img)

def process_dataset(dataset_name, unified_network, device, save_path=None):
    print("\n"*2+"="*80)
    print(f"Processing {dataset_name} dataset:")
    
    ir_dir = os.path.join('datasetname', 'ir')
    vis_dir = os.path.join('datasetname', 'vi')
    
    if not os.path.exists(ir_dir):
        raise ValueError(f"IR directory not found: {ir_dir}")
    if not os.path.exists(vis_dir):
        raise ValueError(f"VIS directory not found: {vis_dir}")
        
    ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.*")))
    vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
    print(f"Found {len(ir_files)} IR images and {len(vis_files)} VIS images")
    print(f"Will save results to: {save_path}")
    
    import processing
    processing._preprocessor_instance = None
    
    test_loader = get_dataloader(
        ir_dir=ir_dir,
        vis_dir=vis_dir,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    unified_network.eval()
    
    with torch.no_grad():
        for i, (vis_img, ir_img, mask, sizes) in enumerate(test_loader):
            try:
                vis_img = vis_img.to(device)
                ir_img = ir_img.to(device)
                
                fused_output, freq_info = unified_network(vis_img, ir_img)
                
                fused_output = fused_output.cpu().numpy()[0, 0]
                
                fused_output = (fused_output * 255).astype(np.float32)
                
                if save_path:
                    try:
                        save_fused_image(fused_output, save_path, original_name)
                    except Exception as e:
                        print(f"Error saving image {original_name}: {str(e)}")
                        
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} images")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    unified_network = nn.DataParallel(UnifiedFusionNetwork()).to(device)
    
    try:
        checkpoint = torch.load(ckpt_path)
        unified_network.load_state_dict(checkpoint['unified_network'])
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading checkpoint:", e)
        print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")
        raise
    
    save_path = os.path.join("fusion_result", "datasetname")
    os.makedirs(save_path, exist_ok=True)
    
    process_dataset("datasetname", unified_network, device, save_path)

if __name__ == "__main__":
    main()