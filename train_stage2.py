# -*- coding: utf-8 -*-
import os
import datetime
import time
import torch
import torch.nn as nn
from processing import get_dataloader
from net import UnifiedFusionNetwork, LFSM
from loss import DetailAwareLoss, MaskFusionL1Loss
import json
import numpy as np
import cv2

# Create save directories
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("results"):
    os.makedirs("results") 

# Set device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration
num_epochs = 80
log_interval = 10
stage1_weights_path = "models/stage1_best_model.pth"  # Path to stage 1 pretrained weights

# Define grid search parameter ranges
param_grid = {
    # Optimizer parameters
    'optimizer_fusion_lr': [0.002],
    'weight_decay_fusion': [0.0001],
    'batch_size': [16],
    
    # MaskFusionL1Loss parameters
    'intensity_weight': [8.0],
    'gradient_weight': [11.0],
    'target_weight': [9.0],
    'back_weight': [5.0],
    
    # DetailAwareLoss parameters
    'lambda_edge': [11.0],
    'lambda_ssim': [5.0],
    'lambda_mask': [9.0],
    'lambda_freq': [3.5],
    'lambda_phase': [1.2]
}

def save_image(fused_output, vis_orig, ir_orig, epoch, batch_idx):
    save_dir = os.path.join("results", f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    
    def tensor_to_numpy(tensor):
        with torch.no_grad():
            img = tensor[0].cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            return img
    
    try:
        vis_orig = tensor_to_numpy(vis_orig)
        ir_orig = tensor_to_numpy(ir_orig)
        fused_img = tensor_to_numpy(fused_output)
        
        top_row = np.concatenate([vis_orig, ir_orig], axis=1)
        bottom_row = np.concatenate([fused_img, fused_img], axis=1)
        combined_img = np.concatenate([top_row, bottom_row], axis=0)
        
        h, w = combined_img.shape[:2]
        cv2.line(combined_img, (w//2, 0), (w//2, h), (0,0,255), 2)
        cv2.line(combined_img, (0, h//2), (w, h//2), (0,0,255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_img, 'Original Visible', (10, 30), font, 1, (255,255,255), 2)
        cv2.putText(combined_img, 'Original IR', (w//2 + 10, 30), font, 1, (255,255,255), 2)
        cv2.putText(combined_img, 'Fused Image', (10, h//2 + 30), font, 1, (255,255,255), 2)
        
        save_path = os.path.join(save_dir, f"batch_{batch_idx}_fused.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"Error saving images: {str(e)}")

def calculate_eta(epoch, batch_idx, total_batches, start_time, num_epochs):
    elapsed_time = time.time() - start_time
    batches_done = epoch * total_batches + batch_idx
    total_batches_to_do = num_epochs * total_batches
    remaining_batches = total_batches_to_do - batches_done
    
    if batches_done == 0:
        return "calculating..."
    
    seconds_per_batch = elapsed_time / batches_done
    seconds_remaining = remaining_batches * seconds_per_batch
    
    return str(datetime.timedelta(seconds=int(seconds_remaining)))

def train_with_params(params):
    print("="*50)
    print("Stage 2: Training Fusion Network")
    print("="*50)
    
    # Initialize data loader
    trainloader = get_dataloader(
        ir_dir="MSRS_train/ir",
        vis_dir="MSRS_train/vi", 
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2
    )
    print(f"Dataset loaded: {len(trainloader)} batches")
    
    # Initialize models
    freq_separator = nn.DataParallel(LFSM()).to(device)
    unified_network = nn.DataParallel(UnifiedFusionNetwork()).to(device)
    
    # Load stage 1 weights
    print(f"Loading Stage 1 weights from {stage1_weights_path}")
    checkpoint = torch.load(stage1_weights_path)
    freq_separator.load_state_dict(checkpoint['freq_separator'])
    
    # Freeze frequency separation module
    freq_separator.eval()
    for param in freq_separator.parameters():
        param.requires_grad = False
    
    # Initialize loss functions
    mask_fusion_l1 = MaskFusionL1Loss(
        intensity_weight=params['intensity_weight'],
        gradient_weight=params['gradient_weight'],
        target_weight=params['target_weight'],
        back_weight=params['back_weight']
    )
    
    fusion_criterion = DetailAwareLoss(
        lambda_edge=params['lambda_edge'],
        lambda_ssim=params['lambda_ssim'],
        lambda_mask=params['lambda_mask'],
        lambda_freq=params['lambda_freq'],
        lambda_phase=params['lambda_phase'],
        mask_fusion_l1=mask_fusion_l1
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        unified_network.parameters(),
        lr=params['optimizer_fusion_lr'],
        weight_decay=params['weight_decay_fusion']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    try:
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            unified_network.train()
            epoch_loss = 0
            valid_batches = 0
            
            for i, (vis_img, ir_img, mask, _) in enumerate(trainloader):
                vis_img = vis_img.to(device)
                ir_img = ir_img.to(device)
                mask = mask.to(device)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    freq_components = freq_separator(vis_img, ir_img)
                
                fused_output, freq_info = unified_network(vis_img, ir_img)
                
                try:
                    fusion_loss, loss_details = fusion_criterion(
                        fused=fused_output,
                        ir=ir_img,
                        vi=vis_img,
                        fused_freq=freq_info,
                        ir_freq=freq_components['ir'],
                        vi_freq=freq_components['rgb'],
                        mask_vi=mask
                    )
                    
                    fusion_loss.backward()
                    torch.nn.utils.clip_grad_norm_(unified_network.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                except Exception as e:
                    print(f"Error in loss computation: {str(e)}")
                    continue
                
                epoch_loss += fusion_loss.item()
                valid_batches += 1
                
                if i % log_interval == 0:
                    eta = calculate_eta(epoch, i, len(trainloader), start_time, num_epochs)
                    print(f"\rEpoch [{epoch+1}/{num_epochs}] "
                          f"Batch [{i}/{len(trainloader)}] "
                          f"Loss: {fusion_loss.item():.6f} "
                          f"ETA: {eta}", end='')
                    
                    # Save example images
                    if i % 20 == 0:
                        save_image(fused_output, vis_img, ir_img, epoch, i)
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                scheduler.step()
                print(f"\nEpoch {epoch+1} completed with average loss: {avg_loss:.6f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    state = {
                        'freq_separator': freq_separator.state_dict(),
                        'unified_network': unified_network.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                        'params': params
                    }
                    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
                    model_path = os.path.join("models", f"stage2_best_{timestamp}.pth")
                    torch.save(state, model_path)
                    print(f"Saved best model to {model_path}")
        
        return best_loss, state
        
    except Exception as e:
        print(f"\nTraining error: {str(e)}")
        return float('inf'), None

def grid_search():
    # Generate all parameter combinations
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    best_loss = float('inf')
    best_params = None
    best_state = None
    
    # Create results log file
    results_file = f"grid_search_results_{datetime.datetime.now().strftime('%m-%d-%H-%M')}.json"
    all_results = []
    
    print(f"Starting grid search with {len(param_combinations)} combinations")
    
    for i, params in enumerate(param_combinations):
        print(f"\nTrying combination {i+1}/{len(param_combinations)}:")
        print(f"Parameters: {params}")
        
        try:
            # Train model and get results
            current_loss, current_state = train_with_params(params)
            
            # Record results
            result = {
                'params': params,
                'loss': float(current_loss),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            all_results.append(result)
            
            # Save current results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=4)
            
            # Update best results
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params
                best_state = current_state
                
                # Save best model
                timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
                best_model_path = os.path.join("models", f"stage2_best_{timestamp}.pth")
                torch.save(best_state, best_model_path)
                print(f"\nNew best model saved: {best_model_path}")
                
            print(f"Current loss: {current_loss:.6f}")
            print(f"Best loss so far: {best_loss:.6f}")
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
    
    return best_params, best_loss, best_state

if __name__ == "__main__":
    try:
        # Set random seed
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.empty_cache()
        
        # Check if stage 1 weights exist
        if not os.path.exists(stage1_weights_path):
            raise FileNotFoundError(f"Stage 1 weights not found at {stage1_weights_path}")
        
        # Execute grid search
        best_params, best_loss, best_state = grid_search()
        
        print("\n" + "="*50)
        print("Grid Search Complete!")
        print("="*50)
        print(f"Best parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Best loss: {best_loss:.6f}")
        
        # Save final best model
        if best_state is not None:
            timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
            final_path = os.path.join("models", f"final_stage2_{timestamp}.pth")
            torch.save(best_state, final_path)
            print(f"\nFinal model saved to: {final_path}")
            
            # Save best parameter configuration
            config_path = os.path.join("models", f"stage2_best_config_{timestamp}.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'best_loss': float(best_loss),
                    'model_path': final_path
                }, f, indent=4)
            print(f"Best configuration saved to: {config_path}")
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        # Save emergency model
        if 'best_state' in locals() and best_state is not None:
            emergency_path = os.path.join("models", f"emergency_stage2_{datetime.datetime.now().strftime('%m-%d-%H-%M')}.pth")
            torch.save(best_state, emergency_path)
            print(f"Emergency model saved to: {emergency_path}")


