# train.py
"""
Main training script for K-Fold Cross-Validation.
Implements:
1. Dynamic class map building.
2. K-Fold CV for 2D and 3D models.
3. Correct train/val transform handling inside K-Fold.
4. Saving CV metrics to 'results/'.
5. Training final models using the 'average best epoch' from CV.

*** OPTIMIZED VERSION ***
- Added Automatic Mixed Precision (AMP) for faster training.
- Set zero_grad(set_to_none=True) for minor speedup.
"""

import torch
import torch.nn as nn
import torch.optim as optim
# ADDED for AMP
from torch.cuda.amp import GradScaler, autocast 

# FIXED (Fix 1): Import Dataset
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset 
import numpy as np
import time
import os
import pandas as pd
from sklearn.model_selection import KFold

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, MODALITY_CONFIG,
    MODEL_2D_CHECKPOINT, MODEL_3D_CHECKPOINT, BATCH_SIZE,
    K_FOLDS, RESULTS_DIR
)
from data.dataset_loader import (
    build_class_map, get_full_dataset, 
    get_2d_transforms, get_mri_transforms
)
from models.vit_model import ViT2DWrapper, ViT3DWrapper
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# FIXED (Fix 3): Helper function for robust logit extraction
def extract_logits(outputs):
    """Extracts logits from various model output types."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs

class TransformSubset(Dataset):
    """
    A wrapper for 2D Subsets that applies a transform on the fly.
    This is used for K-Fold to apply train/val augmentations correctly.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label, modality = self.subset[index]
        image = self.transform(image)
        return image, label, modality
        
    def __len__(self):
        return len(self.subset)

def train_model_fold(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs):
    """
    Trains a model for a single fold and returns:
    (best_val_loss, final_val_accuracy, best_epoch)
    """
    best_val_loss = float('inf')
    best_epoch = 1
    final_val_accuracy = 0.0
    
    # Initialize AMP GradScaler
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in dataloader_train:
            if isinstance(batch, dict): # MONAI (3D)
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            else: # PyTorch (2D)
                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # OPTIMIZED: Use set_to_none=True
            optimizer.zero_grad(set_to_none=True)
            
            # OPTIMIZED: AMP autocast for forward pass
            with autocast(enabled=(DEVICE.type == 'cuda')):
                # FIXED (Fix 3): Use robust logit extraction
                outputs = model(inputs)
                logits = extract_logits(outputs)
                loss = criterion(logits, labels.squeeze(-1).long())

            # OPTIMIZED: Scaler backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
        avg_train_loss = train_loss / len(dataloader_train.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader_val:
                if isinstance(batch, dict):
                    inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                else:
                    inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

                # OPTIMIZED: AMP autocast for validation (no scaler)
                with autocast(enabled=(DEVICE.type == 'cuda')):
                    # FIXED (Fix 3): Use robust logit extraction
                    outputs = model(inputs)
                    logits = extract_logits(outputs)
                    loss = criterion(logits, labels.squeeze(-1).long())
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze(-1)).sum().item()

        avg_val_loss = val_loss / len(dataloader_val.dataset) if total > 0 else 0
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1 # Track the best epoch
            
    final_val_accuracy = val_accuracy # Store final accuracy
    print(f"Fold complete. Best Val Loss: {best_val_loss:.4f} (at Epoch {best_epoch}), Final Val Acc: {final_val_accuracy:.2f}%")
    return best_val_loss, final_val_accuracy, best_epoch

def train_final_model(model, dataloader_train, criterion, optimizer, num_epochs):
    """
    Trains a final model on 100% of the data for a fixed number of epochs.
    No validation.
    """
    model.train()
    
    # Initialize AMP GradScaler
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch in dataloader_train:
            if isinstance(batch, dict): # MONAI (3D)
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            else: # PyTorch (2D)
                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # OPTIMIZED: Use set_to_none=True
            optimizer.zero_grad(set_to_none=True)
            
            # OPTIMIZED: AMP autocast for forward pass
            with autocast(enabled=(DEVICE.type == 'cuda')):
                # FIXED (Fix 3): Use robust logit extraction
                outputs = model(inputs)
                logits = extract_logits(outputs)
                loss = criterion(logits, labels.squeeze(-1).long())
            
            # OPTIMIZED: Scaler backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = train_loss / len(dataloader_train.dataset)
        print(f"Final Training Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")
    
    print("Final training complete.")


def main():
    # ... (rest of main function is correct) ...
    # --- 1. Build Class Map ---
    try:
        class_map = build_class_map()
    except Exception as e:
        print(f"Error building class map: {e}")
        return
        
    num_labels = len(class_map)
    print(f"Found {num_labels} classes: {class_map.keys()}")

    # --- 2. Run K-Fold for 2D Models ---
    print("\n" + "="*50)
    print("STARTING 2D K-FOLD CROSS-VALIDATION")
    print("="*50)
    
    modalities_2d = [m for m, c in MODALITY_CONFIG.items() if c["model_type"] == "2D"]
    full_2d_datasets = [get_full_dataset(mod, class_map) for mod in modalities_2d]
    full_2d_dataset = ConcatDataset(full_2d_datasets)
    print(f"Total 2D samples for K-Fold: {len(full_2d_dataset)}")
    
    avg_best_epoch_2d = EPOCHS # Default

    if len(full_2d_dataset) > K_FOLDS:
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_results_2d = []
        best_epochs_2d = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_2d_dataset)):
            print(f"\n--- 2D FOLD {fold+1}/{K_FOLDS} ---")
            
            # Apply correct transforms to subsets
            train_sub = TransformSubset(Subset(full_2d_dataset, train_ids), get_2d_transforms(is_train=True))
            val_sub = TransformSubset(Subset(full_2d_dataset, val_ids), get_2d_transforms(is_train=False))
            
            dl_train = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            dl_val = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            model_2d_wrapper = ViT2DWrapper(num_labels=num_labels, load_from_scratch=True)
            model_2d = model_2d_wrapper.model
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model_2d.parameters(), lr=LEARNING_RATE)
            
            val_loss, val_acc, best_epoch = train_model_fold(model_2d, dl_train, dl_val, criterion, optimizer, EPOCHS)
            fold_results_2d.append({"fold": fold+1, "val_loss": val_loss, "val_accuracy": val_acc})
            best_epochs_2d.append(best_epoch)

        # --- Save 2D CV Results ---
        cv_df_2d = pd.DataFrame(fold_results_2d)
        cv_df_2d.to_csv(RESULTS_DIR / "cv_results_2d.csv", index=False)
        print("\n--- 2D Cross-Validation Summary ---")
        print(cv_df_2d)
        print(f"Average 2D Val Loss: {cv_df_2d['val_loss'].mean():.4f}")
        print(f"Average 2D Val Accuracy: {cv_df_2d['val_accuracy'].mean():.2f}%")
        
        # Calculate average best epoch
        avg_best_epoch_2d = int(np.mean(best_epochs_2d))
        print(f"Average best epoch: {avg_best_epoch_2d}")

    else:
        print("Not enough 2D data to run K-Fold CV. Skipping.")

    # --- 3. Run K-Fold for 3D Models (MRI) ---
    print("\n" + "="*50)
    print("STARTING 3D K-FOLD CROSS-VALIDATION")
    print("="*50)
    
    modalities_3d = [m for m, c in MODALITY_CONFIG.items() if c["model_type"] == "3D"]
    full_3d_data_dicts = []
    for mod in modalities_3d:
        full_3d_data_dicts.extend(get_full_dataset(mod, class_map))
    print(f"Total 3D samples for K-Fold: {len(full_3d_data_dicts)}")
    
    avg_best_epoch_3d = EPOCHS # Default

    if len(full_3d_data_dicts) > K_FOLDS:
        kfold_3d = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_results_3d = []
        best_epochs_3d = []
        
        # KFold splits indices, so we convert dicts to an array-like
        data_indices = np.arange(len(full_3d_data_dicts))

        for fold, (train_ids, val_ids) in enumerate(kfold_3d.split(data_indices)):
            print(f"\n--- 3D FOLD {fold+1}/{K_FOLDS} ---")
            
            # Create train/val dicts from indices
            train_dicts = [full_3d_data_dicts[i] for i in train_ids]
            val_dicts = [full_3d_data_dicts[i] for i in val_ids]

            # Create MonaiDatasets with correct transforms (Point 3 Fix)
            train_dataset = MonaiDataset(data=train_dicts, transform=get_mri_transforms("MRI", is_train=True))
            val_dataset = MonaiDataset(data=val_dicts, transform=get_mri_transforms("MRI", is_train=False))
            
            dl_train = MonaiDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            dl_val = MonaiDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            model_3d_wrapper = ViT3DWrapper(num_labels=num_labels, load_from_scratch=True)
            model_3d = model_3d_wrapper.model
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model_3d.parameters(), lr=LEARNING_RATE)
            
            val_loss, val_acc, best_epoch = train_model_fold(model_3d, dl_train, dl_val, criterion, optimizer, EPOCHS)
            fold_results_3d.append({"fold": fold+1, "val_loss": val_loss, "val_accuracy": val_acc})
            best_epochs_3d.append(best_epoch)

        # --- Save 3D CV Results ---
        cv_df_3d = pd.DataFrame(fold_results_3d)
        cv_df_3d.to_csv(RESULTS_DIR / "cv_results_3d.csv", index=False)
        print("\n--- 3D Cross-Validation Summary ---")
        print(cv_df_3d)
        print(f"Average 3D Val Loss: {cv_df_3d['val_loss'].mean():.4f}")
        print(f"Average 3D Val Accuracy: {cv_df_3d['val_accuracy'].mean():.2f}%")
        
        # Calculate average best epoch
        avg_best_epoch_3d = int(np.mean(best_epochs_3d))
        print(f"Average best epoch: {avg_best_epoch_3d}")
        
    else:
        print("Not enough 3D data to run K-Fold CV. Skipping.")

    # --- 4. Train Final Models on ALL Data (Point 2 Fix) ---
    print("\n" + "="*50)
    print("TRAINING FINAL DEPLOYABLE MODELS (ON 100% DATA)")
    print("="*50)

    # --- Final 2D Model ---
    if len(full_2d_dataset) > 0:
        print(f"Training final 2D model for {avg_best_epoch_2d} epochs...")
        
        # Apply training transforms to the whole 2D dataset
        final_2d_train_dataset = TransformSubset(full_2d_dataset, get_2d_transforms(is_train=True))
        final_dl_2d = DataLoader(final_2d_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
        model_2d_wrapper = ViT2DWrapper(num_labels=num_labels, load_from_scratch=True)
        model_2d = model_2d_wrapper.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_2d.parameters(), lr=LEARNING_RATE)
        
        train_final_model(model_2d, final_dl_2d, criterion, optimizer, num_epochs=avg_best_epoch_2d)
        
        model_2d_wrapper.save_checkpoint(MODEL_2D_CHECKPOINT)
        print(f"Final 2D model saved to {MODEL_2D_CHECKPOINT}")
    
    # --- Final 3D Model ---
    if len(full_3d_data_dicts) > 0:
        print(f"Training final 3D model for {avg_best_epoch_3d} epochs...")
        
        # Apply training transforms to the whole 3D dataset
        final_3d_train_dataset = MonaiDataset(data=full_3d_data_dicts, transform=get_mri_transforms("MRI", is_train=True))
        final_dl_3d = MonaiDataLoader(final_3d_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
        model_3d_wrapper = ViT3DWrapper(num_labels=num_labels, load_from_scratch=True)
        model_3d = model_3d_wrapper.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_3d.parameters(), lr=LEARNING_RATE)
        
        train_final_model(model_3d, final_dl_3d, criterion, optimizer, num_epochs=avg_best_epoch_3d)
        
        model_3d_wrapper.save_checkpoint(MODEL_3D_CHECKPOINT)
        print(f"Final 3D model saved to {MODEL_3D_CHECKPOINT}")

if __name__ == "__main__":
    main()
