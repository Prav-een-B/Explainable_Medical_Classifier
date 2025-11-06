# train.py
"""
Main training script for K-Fold Cross-Validation.

This script will:
1. Build the dynamic class map from your 'labels.csv' files.
2. Run K-Fold Cross-Validation for the 2D model (XRAY, HISTOPATHOLOGY).
3. Run K-Fold Cross-Validation for the 3D model (MRI).
4. Save the CV performance metrics to the 'results/' folder.
5. Train final 2D and 3D models on ALL data and save them to 'models/weights/'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
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
from data.dataset_loader import build_class_map, get_full_dataset
from models.vit_model import ViT2DWrapper, ViT3DWrapper

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model_fold(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs):
    """
    Trains a model for a single fold and returns the best validation loss.
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for batch in dataloader_train:
            if isinstance(batch, dict): # MONAI (3D)
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            else: # PyTorch (2D)
                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            optimizer.zero_grad()
            
            if "ViT" in model.__class__.__name__: # MONAI ViT
                logits, _ = model(inputs)
            else: # HuggingFace ViT
                logits = model(inputs).logits
            
            loss = criterion(logits, labels.squeeze(-1).long())
            loss.backward()
            optimizer.step()
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

                if "ViT" in model.__class__.__name__:
                    logits, _ = model(inputs)
                else:
                    logits = model(inputs).logits

                loss = criterion(logits, labels.squeeze(-1).long())
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze(-1)).sum().item()

        avg_val_loss = val_loss / len(dataloader_val.dataset)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
    print(f"Fold complete. Best Val Loss: {best_val_loss:.4f}, Final Val Acc: {val_accuracy:.2f}%")
    return best_val_loss, val_accuracy

def main():
    # --- 1. Build Class Map ---
    try:
        class_map = build_class_map()
    except Exception as e:
        print(f"Error building class map: {e}")
        return
        
    num_labels = len(class_map)
    print(f"Found {num_labels} classes: {class_map.keys()}")

    # --- 2. Run K-Fold for 2D Models (XRAY, HISTOPATHOLOGY) ---
    print("\n" + "="*50)
    print("STARTING 2D K-FOLD CROSS-VALIDATION")
    print("="*50)
    
    modalities_2d = [m for m, c in MODALITY_CONFIG.items() if c["model_type"] == "2D"]
    full_2d_datasets = [get_full_dataset(mod, class_map) for mod in modalities_2d]
    full_2d_dataset = ConcatDataset(full_2d_datasets)
    print(f"Total 2D samples for K-Fold: {len(full_2d_dataset)}")

    if len(full_2d_dataset) > K_FOLDS:
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_results_2d = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_2d_dataset)):
            print(f"\n--- 2D FOLD {fold+1}/{K_FOLDS} ---")
            
            # Create subset dataloaders
            train_sub = Subset(full_2d_dataset, train_ids)
            val_sub = Subset(full_2d_dataset, val_ids)
            dl_train = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            dl_val = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            # Initialize a fresh model for this fold
            model_2d_wrapper = ViT2DWrapper(num_labels=num_labels, load_from_scratch=True)
            model_2d = model_2d_wrapper.model
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model_2d.parameters(), lr=LEARNING_RATE)
            
            val_loss, val_acc = train_model_fold(model_2d, dl_train, dl_val, criterion, optimizer, EPOCHS)
            fold_results_2d.append({"fold": fold+1, "val_loss": val_loss, "val_accuracy": val_acc})

        # --- Save 2D CV Results ---
        cv_df_2d = pd.DataFrame(fold_results_2d)
        cv_df_2d.to_csv(RESULTS_DIR / "cv_results_2d.csv", index=False)
        print("\n--- 2D Cross-Validation Summary ---")
        print(cv_df_2d)
        print(f"Average 2D Val Loss: {cv_df_2d['val_loss'].mean():.4f}")
        print(f"Average 2D Val Accuracy: {cv_df_2d['val_accuracy'].mean():.2f}%")

    else:
        print("Not enough 2D data to run K-Fold CV. Skipping.")

    # --- 3. Run K-Fold for 3D Models (MRI) ---
    print("\n" + "="*50)
    print("STARTING 3D K-FOLD CROSS-VALIDATION")
    print("="*50)
    
    modalities_3d = [m for m, c in MODALITY_CONFIG.items() if c["model_type"] == "3D"]
    full_3d_datasets = [get_full_dataset(mod, class_map) for mod in modalities_3d]
    full_3d_dataset = ConcatDataset(full_3d_datasets)
    print(f"Total 3D samples for K-Fold: {len(full_3d_dataset)}")

    if len(full_3d_dataset) > K_FOLDS:
        kfold_3d = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_results_3d = []

        for fold, (train_ids, val_ids) in enumerate(kfold_3d.split(full_3d_dataset)):
            print(f"\n--- 3D FOLD {fold+1}/{K_FOLDS} ---")
            
            train_sub = Subset(full_3d_dataset, train_ids)
            val_sub = Subset(full_3d_dataset, val_ids)
            dl_train = MonaiDataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            dl_val = MonaiDataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            model_3d_wrapper = ViT3DWrapper(num_labels=num_labels, load_from_scratch=True)
            model_3d = model_3d_wrapper.model
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model_3d.parameters(), lr=LEARNING_RATE)
            
            val_loss, val_acc = train_model_fold(model_3d, dl_train, dl_val, criterion, optimizer, EPOCHS)
            fold_results_3d.append({"fold": fold+1, "val_loss": val_loss, "val_accuracy": val_acc})

        # --- Save 3D CV Results ---
        cv_df_3d = pd.DataFrame(fold_results_3d)
        cv_df_3d.to_csv(RESULTS_DIR / "cv_results_3d.csv", index=False)
        print("\n--- 3D Cross-Validation Summary ---")
        print(cv_df_3d)
        print(f"Average 3D Val Loss: {cv_df_3d['val_loss'].mean():.4f}")
        print(f"Average 3D Val Accuracy: {cv_df_3d['val_accuracy'].mean():.2f}%")
        
    else:
        print("Not enough 3D data to run K-Fold CV. Skipping.")

    # --- 4. Train Final Models on ALL Data ---
    print("\n" + "="*50)
    print("TRAINING FINAL DEPLOYABLE MODELS (ON 100% DATA)")
    print("="*50)

    # --- Final 2D Model ---
    if len(full_2d_dataset) > 0:
        print("Training final 2D model...")
        final_dl_2d = DataLoader(full_2d_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        model_2d_wrapper = ViT2DWrapper(num_labels=num_labels, load_from_scratch=True)
        model_2d = model_2d_wrapper.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_2d.parameters(), lr=LEARNING_RATE)
        
        # We re-use train_model_fold but pass an empty validation loader
        dummy_dl_val = DataLoader(Subset(full_2d_dataset, []), batch_size=BATCH_SIZE)
        train_model_fold(model_2d, final_dl_2d, dummy_dl_val, criterion, optimizer, EPOCHS)
        
        model_2d_wrapper.save_checkpoint(MODEL_2D_CHECKPOINT)
        print(f"Final 2D model saved to {MODEL_2D_CHECKPOINT}")
    
    # --- Final 3D Model ---
    if len(full_3d_dataset) > 0:
        print("Training final 3D model...")
        final_dl_3d = MonaiDataLoader(full_3d_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        model_3d_wrapper = ViT3DWrapper(num_labels=num_labels, load_from_scratch=True)
        model_3d = model_3d_wrapper.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_3d.parameters(), lr=LEARNING_RATE)
        
        dummy_dl_val_3d = MonaiDataLoader(Subset(full_3d_dataset, []), batch_size=BATCH_SIZE)
        train_model_fold(model_3d, final_dl_3d, dummy_dl_val_3d, criterion, optimizer, EPOCHS)
        
        model_3d_wrapper.save_checkpoint(MODEL_3D_CHECKPOINT)
        print(f"Final 3D model saved to {MODEL_3D_CHECKPOINT}")

if __name__ == "__main__":
    main()
