# train.py
"""
Main training script for the multimodal medical classifier.

This script will:
1. Build the dynamic class map from your 'labels.csv' files.
2. Train the 2D model (XRAY, HISTOPATHOLOGY).
3. Train the 3D model (MRI).
4. Save the best model weights to the 'models/weights' directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
import time

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, MODALITY_CONFIG,
    MODEL_2D_CHECKPOINT, MODEL_3D_CHECKPOINT, BATCH_SIZE
)
from data.dataset_loader import build_class_map, get_dataloader
from models.vit_model import MultimodalViTWrapper

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """Generic training loop for a single model."""
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for batch in dataloader["train"]:
            # Dataloader returns (data, label, modality)
            # We ignore modality, as this loop is for one model type
            if isinstance(batch, dict): # MONAI (3D)
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            else: # PyTorch (2D)
                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, nn.Module): # Standard torch module
                if "ViT" in model.__class__.__name__: # MONAI ViT
                    logits, _ = model(inputs)
                else: # HuggingFace ViT
                    logits = model(inputs).logits
            
            loss = criterion(logits, labels.squeeze(-1).long()) # Ensure labels are 1D long
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        avg_train_loss = train_loss / len(dataloader["train"].dataset)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader["val"]:
                if isinstance(batch, dict): # MONAI (3D)
                    inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                else: # PyTorch (2D)
                    inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

                if "ViT" in model.__class__.__name__: # MONAI ViT
                    logits, _ = model(inputs)
                else: # HuggingFace ViT
                    logits = model(inputs).logits

                loss = criterion(logits, labels.squeeze(-1).long())
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(dataloader["val"].dataset)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}. Saving model...")
            # This is a generic loop, we return the best state dict
            best_model_state = model.state_dict()
            
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    return best_model_state

def main():
    # --- 1. Build Class Map ---
    # This MUST run first. It scans your data and creates the class_map.json
    try:
        class_map = build_class_map()
    except Exception as e:
        print(f"Error building class map: {e}")
        print("Please ensure your 'data/raw/<modality>/<split>/labels.csv' files are correct.")
        return
        
    num_labels = len(class_map)
    print(f"Found {num_labels} classes: {class_map.keys()}")

    # --- 2. Initialize Models for Training ---
    # We pass load_from_scratch=True to get base models
    wrapper = MultimodalViTWrapper(num_labels=num_labels, load_from_scratch=True)
    
    # --- 3. Prepare 2D Data ---
    print("\n--- Preparing 2D Datasets (XRAY, HISTOPATHOLOGY) ---")
    modalities_2d = [m for m, c in MODALITY_CONFIG.items() if c["model_type"] == "2D"]
    datasets_2d_train = []
    datasets_2d_val = []
    
    for mod in modalities_2d:
        try:
            datasets_2d_train.append(get_dataloader(mod, class_map, "train").dataset)
            datasets_2d_val.append(get_dataloader(mod, class_map, "test").dataset) # Using 'test' as val
        except Exception as e:
            print(f"Could not load 2D modality {mod}: {e}")
            
    if not datasets_2d_train:
        print("No 2D data found. Skipping 2D model training.")
    else:
        # Combine all 2D datasets to train the single 2D model
        concat_dataset_train = ConcatDataset(datasets_2d_train)
        concat_dataset_val = ConcatDataset(datasets_2d_val)
        
        dataloaders_2d = {
            "train": DataLoader(concat_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
            "val": DataLoader(concat_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        }
        print(f"Total 2D training samples: {len(concat_dataset_train)}")
        print(f"Total 2D validation samples: {len(concat_dataset_val)}")
        
        # --- 4. Train 2D Model ---
        print("\n--- Training 2D Model ---")
        model_2d_wrapper = wrapper.get_wrapper("XRAY") # Get the 2D wrapper
        model_2d = model_2d_wrapper.model
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_2d.parameters(), lr=LEARNING_RATE)
        
        best_2d_state = train_model(model_2d, dataloaders_2d, criterion, optimizer, EPOCHS)
        
        # Save the final best model
        model_2d.load_state_dict(best_2d_state)
        model_2d_wrapper.save_checkpoint(MODEL_2D_CHECKPOINT)
        print("Best 2D model saved.")

    # --- 5. Prepare 3D Data ---
    print("\n--- Preparing 3D Datasets (MRI) ---")
    modalities_3d = [m for m, c in MODALITY_CONFIG.items() if c["model_type"] == "3D"]
    datasets_3d_train = []
    datasets_3d_val = []

    for mod in modalities_3d:
        try:
            datasets_3d_train.append(get_dataloader(mod, class_map, "train").dataset)
            datasets_3d_val.append(get_dataloader(mod, class_map, "test").dataset)
        except Exception as e:
            print(f"Could not load 3D modality {mod}: {e}")

    if not datasets_3d_train:
        print("No 3D data found. Skipping 3D model training.")
    else:
        # Combine all 3D datasets
        concat_dataset_3d_train = ConcatDataset(datasets_3d_train)
        concat_dataset_3d_val = ConcatDataset(datasets_3d_val)

        dataloaders_3d = {
            "train": MonaiDataLoader(concat_dataset_3d_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
            "val": MonaiDataLoader(concat_dataset_3d_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        }
        print(f"Total 3D training samples: {len(concat_dataset_3d_train)}")
        print(f"Total 3D validation samples: {len(concat_dataset_3d_val)}")

        # --- 6. Train 3D Model ---
        print("\n--- Training 3D Model ---")
        model_3d_wrapper = wrapper.get_wrapper("MRI") # Get the 3D wrapper
        model_3d = model_3d_wrapper.model
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_3d.parameters(), lr=LEARNING_RATE)
        
        best_3d_state = train_model(model_3d, dataloaders_3d, criterion, optimizer, EPOCHS)

        # Save the final best model
        model_3d.load_state_dot(best_3d_state)
        model_3d_wrapper.save_checkpoint(MODEL_3D_CHECKPOINT)
        print("Best 3D model saved.")
        
    print("\n--- All Training Finished ---")

if __name__ == "__main__":
    main()
