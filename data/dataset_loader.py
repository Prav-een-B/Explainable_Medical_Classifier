# data/dataset_loader.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np
from PIL import Image
import glob
import os
import pandas as pd
import json

from config import (
    MODALITY_CONFIG, IMG_SIZE_2D, BATCH_SIZE, 
    DATA_ROOT, # <-- KAGGLE: This is now a DICT
    CLASS_MAP_JSON, NUM_WORKERS
)

# --- MONAI Imports ---
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandAffined, ToTensord
)

# --- Helper for NIH/Multi-label ---
def parse_label(raw_label_str):
    """
    Parses a label string. For NIH, it might be 'A|B|C'.
    We'll take the first label for this multi-class setup.
    """
    if pd.isna(raw_label_str):
        return "Unknown" # Handle missing labels
    
    # Take the first finding as the class
    first_label = str(raw_label_str).split('|')[0]
    
    # 'No Finding' is a valid class
    if first_label == "No Finding":
        return "No Finding"
        
    # Filter out 'No Finding' if other findings exist
    if "|" in raw_label_str:
        labels = str(raw_label_str).split('|')
        first_finding = next((l for l in labels if l != "No Finding"), "No Finding")
        return first_finding
        
    return first_label

# --- Dynamic Class Map Builder ---
def build_class_map():
    print("Building class map...")
    all_disease_names = set()
    
    # KAGGLE: DATA_ROOT is now a dict. Iterate over its keys.
    for modality in MODALITY_CONFIG.keys():
        if modality not in DATA_ROOT:
            print(f"Warning: No data path in config.py for {modality}. Skipping.")
            continue
        
        modality_root = DATA_ROOT[modality]
        
        for split in ["train", "test"]:
            # --- KAGGLE NIH XRAY SPECIAL HANDLING ---
            if modality == "XRAY":
                # Assumes you will create 'train_labels.csv' and 'test_labels.csv'
                # from the main NIH file.
                labels_path = modality_root / f"{split}_labels.csv"
                
                if not labels_path.exists():
                   # Fallback for user's uploaded 'sample_labels.csv'
                   if split == 'train': # Only load it once
                        labels_path = modality_root / "sample_labels.csv"
                        print(f"Info: Using '{labels_path}' for XRAY class map.")
                   else:
                       continue # Don't load sample twice
            
            else:
                # Assumed structure for other modalities
                labels_path = modality_root / split / "labels.csv"
            
            if not labels_path.exists():
                print(f"Warning: Missing required file {labels_path}")
                continue
                
            df = pd.read_csv(labels_path)
            
            # KAGGLE: Determine correct column
            if modality == "XRAY":
                label_col = "Finding Labels"
            else: # Histo, MRI
                label_col = "disease"
                
            if label_col not in df.columns:
                print(f"Warning: '{label_col}' column not found in {labels_path}. Trying 'disease'.")
                if "disease" in df.columns:
                    label_col = "disease"
                else:
                    raise ValueError(f"'{label_col}' (or 'disease') column not found in {labels_path}")

            # Parse all labels (handles multi-label split)
            for raw_label in df[label_col]:
                all_disease_names.add(parse_label(raw_label))
    
    if not all_disease_names:
        raise FileNotFoundError(f"No label files found in paths specified in {DATA_ROOT}. Cannot build class map.")
    
    sorted_names = sorted(list(all_disease_names))
    class_map = {name: i for i, name in enumerate(sorted_names)}
    
    # Ensure output dir exists
    os.makedirs(CLASS_MAP_JSON.parent, exist_ok=True)
    with open(CLASS_MAP_JSON, 'w') as f:
        json.dump(class_map, f, indent=4)
    print(f"Class map built and saved to {CLASS_MAP_JSON}")
    print(f"Classes found: {class_map}")
    return class_map

def load_class_map():
    # Path is now /kaggle/working/data/class_map.json
    if not CLASS_MAP_JSON.exists():
        raise FileNotFoundError(f"{CLASS_MAP_JSON} not found. Please run train.py first to build the class map.")
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
    return class_map

# --- 2D Dataset Classes ---
class Base2DDataset(Dataset):
    # KAGGLE: This class needs to handle the NIH format
    def __init__(self, data_dir, csv_path, transform, class_map, modality, img_col, label_col):
        self.data_dir = data_dir # This should be the dir containing images
        self.transform = transform
        self.class_map = class_map
        self.modality = modality
        self.img_col = img_col
        self.label_col = label_col
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Required {csv_path} not found.")
        self.label_df = pd.read_csv(csv_path)
        
        self.image_paths = []
        self.labels = []
        
        for _, row in self.label_df.iterrows():
            img_path = os.path.join(self.data_dir, row[self.img_col])
            
            if not os.path.exists(img_path):
                # Try to find it (e.g., in 'images' subdir)
                img_path_alt = os.path.join(self.data_dir, "images", row[self.img_col])
                if os.path.exists(img_path_alt):
                    img_path = img_path_alt
                else:
                    print(f"Warning: Image {img_path} (or alt) listed in CSV but not found.")
                    continue
                    
            self.image_paths.append(img_path)
            
            # KAGGLE: Use the parsing function
            parsed_label = parse_label(row[self.label_col])
            if parsed_label not in self.class_map:
                 print(f"Warning: Label '{parsed_label}' not in class_map. Skipping.")
                 continue
            self.labels.append(self.class_map[parsed_label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), self.modality

def get_2d_transforms(is_train=True):
    # ... (function is correct) ...
    transform_list = [
        transforms.Resize((IMG_SIZE_2D, IMG_SIZE_2D)),
    ]
    if is_train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(transform_list)

# --- 3D Dataset (NIfTI Loader for MRI) ---
def get_mri_transforms(modality, is_train=True):
    # ... (function is correct) ...
    cfg = MODALITY_CONFIG[modality]
    img_size_3d = cfg["size"]
    transform_list = [
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=img_size_3d),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=2000.0, 
            b_min=0.0, b_max=1.0, clip=True
        ),
    ]
    if is_train:
        transform_list.append(
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1)
            )
        )
    transform_list.append(ToTensord(keys=["image", "label"]))
    return Compose(transform_list)


# --- Main Loader Function (for Inference) ---
def get_dataloader(modality, class_map, split="test", batch_size=BATCH_SIZE, shuffle=True):
    is_train = (split == "train")
    
    # KAGGLE: Get root path for this modality
    modality_root = DATA_ROOT[modality]

    if modality in ["XRAY", "HISTOPATHOLOGY"]:
        
        # KAGGLE: Handle XRAY paths and columns
        if modality == "XRAY":
            # Assumes you have 'train_labels.csv' and 'test_labels.csv'
            # And images are in 'modality_root' or 'modality_root/images'
            csv_path = modality_root / f"{split}_labels.csv"
            img_dir = modality_root # The Base2DDataset will check /images subdir
            img_col = "Image Index"
            label_col = "Finding Labels"
            
            if not csv_path.exists():
                if split == 'test':
                     # In case of sample_labels.csv, use it for 'test' demo
                     csv_path = modality_root / "sample_labels.csv"
                     print(f"Info: Using '{csv_path}' for XRAY test dataloader.")
                else:
                     raise FileNotFoundError(f"Missing {csv_path}")
        else:
            # Assumed default for HISTO
            data_dir = modality_root / split
            csv_path = data_dir / "labels.csv"
            img_dir = data_dir
            img_col = "image_filename"
            label_col = "disease"

        transforms_2d = get_2d_transforms(is_train=is_train)
        dataset = Base2DDataset(img_dir, csv_path, 
                                transform=transforms_2d, 
                                class_map=class_map, 
                                modality=modality,
                                img_col=img_col,
                                label_col=label_col)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)
    
    elif modality == "MRI":
        data_dir = modality_root / split
        labels_path = data_dir / "labels.csv"
        img_col = "image_filename"
        label_col = "disease"
        
        if not labels_path.exists():
            raise FileNotFoundError(f"Required {labels_path} not found.")
        label_df = pd.read_csv(labels_path)
        data_dicts = []
        for _, row in label_df.iterrows():
            img_path = str(data_dir / row[img_col])
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} listed in CSV but not found.")
                continue
            
            # KAGGLE: Use parsing function for consistency
            parsed_label = parse_label(row[label_col])
            if parsed_label not in class_map:
                 print(f"Warning: Label '{parsed_label}' not in class_map. Skipping.")
                 continue
            
            data_dicts.append({
                "image": img_path,
                "label": class_map[parsed_label]
            })
        if not data_dicts:
            print(f"Warning: No valid NIfTI files found for {modality} in {data_dir}.")
        transforms_3d = get_mri_transforms(modality, is_train=is_train)
        dataset = MonaiDataset(data=data_dicts, transform=transforms_3d)
        return MonaiDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)
    
    else:
        raise ValueError(f"Modality {modality} not supported.")

# --- K-Fold Cross-Validation Function ---
def get_full_dataset(modality, class_map):
    """
    Loads and combines ALL data (train and test) for a single modality
    into one single Dataset object (for 2D) or list of dicts (for 3D).
    """
    print(f"Loading full dataset for {modality}...")
    
    # KAGGLE: Get root path for this modality
    modality_root = DATA_ROOT[modality]
    
    if modality in ["XRAY", "HISTOPATHOLOGY"]:
        
        # FIXED (Fix 7): This dataset now stores paths and loads images lazily
        class CustomPILDataset(Dataset):
            def __init__(self, data_list):
                self.data_list = data_list 
            def __len__(self):
                return len(self.data_list)
            def __getitem__(self, idx):
                img_path, label, modality = self.data_list[idx]
                image = Image.open(img_path).convert("RGB")
                return image, torch.tensor(label, dtype=torch.long), modality

        data_list = []
        
        # KAGGLE: Handle XRAY paths and columns
        if modality == "XRAY":
            split_csvs = {
                "train": modality_root / "train_labels.csv",
                "test": modality_root / "test_labels.csv"
            }
            img_dir = modality_root 
            img_col = "Image Index"
            label_col = "Finding Labels"
            
            # Fallback for sample_labels.csv
            if not split_csvs["train"].exists() and not split_csvs["test"].exists():
                print(f"Warning: No train/test CSVs found. Using '{modality_root / 'sample_labels.csv'}' for XRAY.")
                split_csvs = {"train": modality_root / "sample_labels.csv"} # Just use sample
        else:
            # Assumed default for HISTO
            split_csvs = {
                "train": modality_root / "train" / "labels.csv",
                "test": modality_root / "test" / "labels.csv"
            }
            img_dirs = {
                 "train": modality_root / "train",
                 "test": modality_root / "test"
            }
            img_col = "image_filename"
            label_col = "disease"

        
        for split, csv_path in split_csvs.items():
            try:
                if not csv_path.exists():
                    print(f"Info: No K-Fold data for {modality} {split} ({csv_path})")
                    continue
                
                label_df = pd.read_csv(csv_path)
                
                # Set the correct image directory
                if modality == "XRAY":
                    current_img_dir = img_dir
                else:
                    current_img_dir = img_dirs[split]
                
                for _, row in label_df.iterrows():
                    img_path = os.path.join(current_img_dir, row[img_col])
                    
                    if not os.path.exists(img_path):
                        img_path_alt = os.path.join(current_img_dir, "images", row[img_col])
                        if os.path.exists(img_path_alt):
                            img_path = img_path_alt
                        else:
                            continue # Skip if not found

                    # KAGGLE: Use parsing function
                    parsed_label = parse_label(row[label_col])
                    if parsed_label not in class_map:
                         continue
                    
                    label = class_map[parsed_label]
                    data_list.append((img_path, label, modality))
                    
            except Exception as e:
                print(f"Could not load {split} data for {modality}: {e}")
                
        return CustomPILDataset(data_list)

    elif modality == "MRI":
        data_dicts = []
        img_col = "image_filename"
        label_col = "disease"
        
        for split in ["train", "test"]:
            try:
                data_dir = modality_root / split
                labels_path = data_dir / "labels.csv"
                if not labels_path.exists():
                    continue
                label_df = pd.read_csv(labels_path)
                
                for _, row in label_df.iterrows():
                    img_path = str(data_dir / row[img_col])
                    if not os.path.exists(img_path):
                        continue
                    
                    parsed_label = parse_label(row[label_col])
                    if parsed_label not in class_map:
                         continue
                    
                    data_dicts.append({
                        "image": img_path,
                        "label": class_map[parsed_label]
                    })
            except Exception as e:
                print(f"Could not load {split} data for {modality}: {e}")
                
        return data_dicts # Return list of dicts
