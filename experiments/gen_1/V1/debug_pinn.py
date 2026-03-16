#!/usr/bin/env python3
"""
Debug script to test the fixed PINN training.
Runs a few epochs with the stability improvements.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from architectures import PINN
from train import load_data, split_data, create_dataloaders

def debug_pinn_training():
    """Run PINN training with debug output."""
    
    print("=" * 80)
    print("PINN TRAINING DEBUG - Running a few actual training steps")
    print("=" * 80)
    
    # Configuration
    config = {
        'data_path': '/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/data_generation/data/training_50M.npz',
        'train_fraction': 0.8,
        'val_fraction': 0.1,
        'test_fraction': 0.1,
        'max_samples': 10000,  # Small for debugging
        'model_type': 'pinn',
        'hidden_dims': [64, 64],  # Small for speed
        'activation': 'silu',
        'lambda_pde': 0.1,  # Start smaller
        'lambda_ic': 0.1,
        'n_collocation': 3,  # Reduced for debugging
        'batch_size': 256,
        'learning_rate': 1e-4,  # Smaller LR
        'device': 'cpu',
        'physics_warmup_epochs': 5,
        'grad_clip': 1.0,
        'num_workers': 0,
        'pin_memory': False,
    }
    
    print(f"\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    # Load data
    print("\n" + "-" * 80)
    print("Loading data...")
    X, Y, P = load_data(config)
    print(f"Data loaded: X={X.shape}, Y={Y.shape}, P={P.shape}")
    
    # Split and create dataloaders
    splits = split_data(X, Y, P, config)
    loaders = create_dataloaders(splits, config)
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating PINN model...")
    device = torch.device(config['device'])
    model = PINN(
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        lambda_pde=config['lambda_pde'],
        lambda_ic=config['lambda_ic'],
        n_collocation=config['n_collocation']
    ).to(device)
    
    # Set normalization using train data
    X_train, Y_train, P_train = splits['train']
    model.set_normalization(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(Y_train).to(device)
    )
    
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input normalization - mean: {model.input_mean[:4]}, std: {model.input_std[:4]}")
    print(f"Output normalization - mean: {model.output_mean}, std: {model.output_std}")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    print("\n" + "-" * 80)
    print("Training for 20 epochs...")
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        total_data_loss = 0
        total_physics_loss = 0
        n_batches = 0
        n_skipped = 0
        
        # Physics warmup
        physics_scale = min(1.0, epoch / config['physics_warmup_epochs'])
        
        for batch_idx, (x, y, p) in enumerate(loaders['train']):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x)
            
            # Data loss
            data_loss = criterion(y_pred, y)
            
            if torch.isnan(data_loss) or torch.isinf(data_loss):
                n_skipped += 1
                continue
            
            # Physics loss
            physics_losses = model.compute_physics_loss(x, y_pred)
            physics_loss = sum(physics_losses.values())
            
            if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                physics_loss = torch.tensor(0.0, device=device)
            
            loss = data_loss + physics_scale * physics_loss
            
            # Check final loss
            if torch.isnan(loss) or torch.isinf(loss):
                n_skipped += 1
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
            n_batches += 1
            
            # Debug first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"\n  First batch details:")
                print(f"    y_pred range: [{y_pred.min().item():.2f}, {y_pred.max().item():.2f}]")
                print(f"    y_true range: [{y.min().item():.2f}, {y.max().item():.2f}]")
                print(f"    data_loss: {data_loss.item():.6e}")
                print(f"    physics_loss (IC): {physics_losses['ic'].item():.6e}")
                print(f"    physics_loss (PDE): {physics_losses['pde'].item():.6e}")
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            avg_data = total_data_loss / n_batches
            avg_phys = total_physics_loss / n_batches
            print(f"Epoch {epoch+1}: loss={avg_loss:.6e}, data={avg_data:.6e}, phys={avg_phys:.6e}, "
                  f"scale={physics_scale:.2f}, skipped={n_skipped}")
        else:
            print(f"Epoch {epoch+1}: ALL BATCHES SKIPPED!")
    
    # Final evaluation
    print("\n" + "-" * 80)
    print("Final evaluation on validation set...")
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for x, y, p in loaders['val']:
            x = x.to(device)
            y_pred = model(x)
            val_preds.append(y_pred.cpu())
            val_targets.append(y)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        
        pos_err = torch.sqrt((val_preds[:, 0] - val_targets[:, 0])**2 + 
                            (val_preds[:, 1] - val_targets[:, 1])**2)
        
        print(f"Position error: {pos_err.mean().item():.4f} ± {pos_err.std().item():.4f} mm")
        print(f"Position 95%: {torch.quantile(pos_err, 0.95).item():.4f} mm")
    
    print("\n" + "=" * 80)
    print("Debug complete!")
    print("=" * 80)

if __name__ == '__main__':
    debug_pinn_training()
