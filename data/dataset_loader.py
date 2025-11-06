# data/dataset_loader.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import medmnist
import numpy as np
from PIL import Image
from config import MODALITY_CONFIG, IMG_SIZE, BATCH_SIZE

# --- Base and Mock Dataset Classes ---
class BaseMedicalDataset(Dataset):
    def __init__(self, data_list, labels, modality):
        self.data_list = data_list
        self.labels = labels
        self.modality = modality

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Return image/volume, label, and modality type
        return self.data_list[idx], self.labels[idx], self.modality
    
class Mock3DDataset(BaseMedicalDataset):
    """Mocks a 3D volumetric dataset for CT/MRI (Output: Tensor D, H, W)."""
    def __init__(self, num_samples, modality):
        cfg = MODALITY_CONFIG[modality]
        
        # Generate random 3D data (Depth, H, W)
        data_list = [
            torch.rand(cfg["depth"], cfg["size"], cfg["size"])
            for _ in range(num_samples)
        ]
        labels = torch.randint(0, 2, (num_samples, 1)).float()
        super().__init__(data_list, labels, modality)
        
    def __getitem__(self, idx):
        volume = self.data_list[idx]
        return volume, self.labels[idx], self.modality

class Medmnist2DDataset(BaseMedicalDataset):
    """Wrapper for medmnist, used to mock XRAY/ULTRASOUND (Output: PIL Image)."""
    def __init__(self, dataset_name, split, modality):
        DataClass = getattr(medmnist, f"{dataset_name}MNIST")
        
        # Define 2D transforms: resize to ViT standard, convert to Tensor, normalize
        transform_list = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(), 
        ]
        transform = transforms.Compose(transform_list)

        data_obj = DataClass(split=split, transform=transform, download=True)
        
        data_list = [data_obj.imgs[i] for i in range(len(data_obj))]
        labels = torch.from_numpy(data_obj.labels).float()

        super().__init__(data_list, labels, modality)
        
# --- Main Loader Function ---
def get_dataloader(dataset_name, modality, split="test", batch_size=BATCH_SIZE, shuffle=False):
    if modality in ["XRAY", "ULTRASOUND"]:
        dataset = Medmnist2DDataset(dataset_name, split, modality)
    elif modality in ["CT", "MRI"]:
        print(f"Using mock 3D data for {modality}")
        dataset = Mock3DDataset(num_samples=batch_size * 5, modality=modality)
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
