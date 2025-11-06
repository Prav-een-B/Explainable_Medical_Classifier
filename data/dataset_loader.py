# data/dataset_loader.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import medmnist
import numpy as np
from PIL import Image
import glob
import os

from config import MODALITY_CONFIG, IMG_SIZE_2D, BATCH_SIZE, DATA_ROOT

# --- MONAI Imports for 3D Data Loading ---
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)

# --- 2D Dataset (Original) ---
class Medmnist2DDataset(Dataset):
    """Wrapper for medmnist, used to mock XRAY/ULTRASOUND."""
    def __init__(self, dataset_name, split, modality):
        DataClass = getattr(medmnist, f"{dataset_name}MNIST")
        
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE_2D, IMG_SIZE_2D)),
            transforms.ToTensor(), 
        ])

        data_obj = DataClass(split=split, transform=transform, download=True)
        
        self.data_list = [data_obj.imgs[i] for i in range(len(data_obj))]
        self.labels = torch.from_numpy(data_obj.labels).float()
        self.modality = modality

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Medmnist images are PIL, we apply transform
        img = self.data_list[idx]
        if self.transform:
            img = self.transform(img)
        # Return image tensor, label, and modality type
        return img, self.labels[idx], self.modality

# --- 3D Dataset (Replaced Mock with Real NIfTI Loader) ---
def get_3d_transforms(modality):
    """Returns MONAI transform pipeline for 3D volumes."""
    cfg = MODALITY_CONFIG[modality]
    img_size_3d = cfg["size"]

    # Define intensity scaling
    # These values are typical for CT scans (soft tissue window)
    # They would need to be changed for MRI
    intensity_min = -1024.0
    intensity_max = 3072.0
    
    return Compose(
        [
            # Load NIfTI file, "image" key points to the filepath
            LoadImaged(keys=["image"]), 
            # Ensure shape is (C, D, H, W)
            EnsureChannelFirstd(keys=["image"]),
            # Resize to model's expected input size
            Resized(keys=["image"], spatial_size=img_size_3d),
            # Normalize pixel intensities
            ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_min,
                a_max=intensity_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Convert to Tensor
            ToTensord(keys=["image", "label"]),
        ]
    )

def get_dataloader(dataset_name, modality, split="test", batch_size=BATCH_SIZE, shuffle=False):
    if modality in ["XRAY", "ULTRASOUND"]:
        dataset = Medmnist2DDataset(dataset_name, split, modality)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    elif modality in ["CT", "MRI"]:
        # --- Real NIfTI Data Loading ---
        # We assume data is in: data/raw/<modality>/<split>/
        data_dir = DATA_ROOT / modality / split
        
        # NOTE: Create dummy NIfTI files if they don't exist for demo
        _create_dummy_nifti_files(data_dir, 5)

        # Scan for NIfTI files
        image_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
        
        # Create dummy labels (replace with real labels)
        labels = [np.random.randint(0, 2) for _ in image_files]
        
        # Create a list of dictionaries for MONAI
        data_dicts = [
            {"image": img_path, "label": label}
            for img_path, label in zip(image_files, labels)
        ]

        if not data_dicts:
            print(f"Warning: No NIfTI files found in {data_dir}. Demo will fail for {modality}.")

        transforms_3d = get_3d_transforms(modality)
        dataset = MonaiDataset(data=data_dicts, transform=transforms_3d)
        
        # Use MONAI's DataLoader
        return MonaiDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    else:
        raise ValueError(f"Modality {modality} not supported.")

def _create_dummy_nifti_files(data_dir, num_files=5):
    """Helper to create placeholder NIfTI files for the demo."""
    import nibabel as nib
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_files):
        filepath = os.path.join(data_dir, f"dummy_scan_{i}.nii.gz")
        if not os.path.exists(filepath):
            print(f"Creating dummy NIfTI file: {filepath}")
            # Create a 1x1x1 dummy volume
            dummy_data = np.zeros((1, 1, 1), dtype=np.int16)
            img = nib.Nifti1Image(dummy_data, affine=np.eye(4))
            nib.save(img, filepath)
