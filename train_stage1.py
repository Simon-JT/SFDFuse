import os
import datetime
import time
import torch
import torch.nn as nn
from processing import get_dataloader
from net import LFSM
from loss import FrequencySeparationLoss
import json
from itertools import product

# Create model save directory
if not os.path.exists("models"):
    os.makedirs("models")

# Set device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration
num_epochs = 40
log_interval = 10

# Define grid search parameter ranges
param_grid = {
    # Optimizer parameters
    'learning_rate': [3e-5, 5e-5, 7e-5],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'batch_size': [16],
    
    # FrequencySeparationLoss parameters
    'high_freq_weight': [0.005, 0.01, 0.02],  # HighFrequencyEnergyLoss weight
    'contrast_weight': [0.005, 0.01, 0.02],   # FrequencyDomainContrastLoss weight
    'rgb_weight': [0.8, 1.0, 1.2],           # RGB branch weight
    'ir_weight': [0.8, 1.0, 1.2]             # IR branch weight
}

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

def train_stage1(params):
    print("="*50)
    print("Stage 1: Training Frequency Separation Module")
    print("="*50)
    print("Current parameters:", params)
    
    # Initialize data loader
    trainloader = get_dataloader(
        ir_dir="MSRS_train/ir",
        vis_dir="MSRS_train/vi", 
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2
    )
    print(f"Dataset loaded: {len(trainloader)} batches")
    
    # Initialize frequency separation module
    freq_separator = nn.DataParallel(LFSM()).to(device)
    
    # Initialize frequency separation loss with weights
    freq_criterion = FrequencySeparationLoss()
    freq_criterion.high_freq_loss.weight = params['high_freq_weight']
    freq_criterion.contrast_loss.weight = params['contrast_weight']
    freq_criterion.rgb_weight = params['rgb_weight']
    freq_criterion.ir_weight = params['ir_weight']
    freq_criterion = freq_criterion.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        freq_separator.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
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
            freq_separator.train()
            epoch_loss = 0
            valid_batches = 0
            
            for i, (vis_img, ir_img, mask, _) in enumerate(trainloader):
                vis_img = vis_img.to(device)
                ir_img = ir_img.to(device)
                
                optimizer.zero_grad()
                
                # Get frequency separation results
                freq_components = freq_separator(vis_img, ir_img)
                
                # Calculate frequency separation loss
                freq_loss = freq_criterion(
                    freq_components['rgb']['high_freq'],
                    freq_components['rgb']['low_freq'],
                    freq_components['ir']['high_freq'],
                    freq_components['ir']['low_freq']
                )
                
                freq_loss.backward()
                torch.nn.utils.clip_grad_norm_(freq_separator.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += freq_loss.item()
                valid_batches += 1
                
                if i % log_interval == 0:
                    eta = calculate_eta(epoch, i, len(trainloader), start_time, num_epochs)
                    print(f"\rEpoch [{epoch+1}/{num_epochs}] "
                          f"Batch [{i}/{len(trainloader)}] "
                          f"Loss: {freq_loss.item():.6f} "
                          f"ETA: {eta}", end='')
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                scheduler.step()
                print(f"\nEpoch {epoch+1} completed with average loss: {avg_loss:.6f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    state = {
                        'freq_separator': freq_separator.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                        'params': params,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
                    model_path = os.path.join("models", f"stage1_best_{timestamp}.pth")
                    torch.save(state, model_path)
                    print(f"Saved best model to {model_path}")
        
        return best_loss, state, model_path
        
    except Exception as e:
        print(f"\nTraining error: {str(e)}")
        return float('inf'), None, None

def grid_search():
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    best_loss = float('inf')
    best_params = None
    best_state = None
    best_model_path = None
    
    # Create results log file
    results_file = f"stage1_grid_search_results_{datetime.datetime.now().strftime('%m-%d-%H-%M')}.json"
    all_results = []
    
    print(f"Starting grid search with {len(param_combinations)} combinations")
    
    for i, params in enumerate(param_combinations):
        print(f"\nTrying combination {i+1}/{len(param_combinations)}:")
        print(f"Parameters: {params}")
        
        try:
            # Train model and get results
            current_loss, current_state, model_path = train_stage1(params)
            
            # Record results
            result = {
                'params': params,
                'loss': float(current_loss),
                'model_path': model_path,
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
                best_model_path = model_path
                print(f"\nNew best model found!")
                
            print(f"Current loss: {current_loss:.6f}")
            print(f"Best loss so far: {best_loss:.6f}")
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
    
    return best_params, best_loss, best_state, best_model_path

if __name__ == "__main__":
    try:
        # Set random seed
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.empty_cache()
        
        # Execute grid search
        best_params, best_loss, best_state, best_model_path = grid_search()
        
        print("\n" + "="*50)
        print("Grid Search Complete!")
        print("="*50)
        print(f"Best parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Best model saved to: {best_model_path}")
        
        # Save best parameter configuration
        if best_state is not None:
            config_path = os.path.join("models", f"stage1_best_config_{datetime.datetime.now().strftime('%m-%d-%H-%M')}.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'best_loss': float(best_loss),
                    'model_path': best_model_path
                }, f, indent=4)
            print(f"Best configuration saved to: {config_path}")
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        # Save emergency model
        if 'best_state' in locals() and best_state is not None:
            emergency_path = os.path.join("models", f"emergency_stage1_{datetime.datetime.now().strftime('%m-%d-%H-%M')}.pth")
            torch.save(best_state, emergency_path)
            print(f"Emergency model saved to: {emergency_path}") 