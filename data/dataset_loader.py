# ============================================================
# dataset_loader.py — unified loader for XRay, Skin, and MRI
# ============================================================

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms

# --- Config ---
IMG_SIZE = 224

KAGGLE_PATHS = {
    "XRAY": Path("/kaggle/input/data"),
    "SKIN": Path("/kaggle/input/skin-cancer-mnist-ham10000"),
    "MRI":  Path("/kaggle/input/brain-mri-images-for-brain-tumor-detection")
}

# --- Transforms ---
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================================================
# NIH Chest X-ray Dataset
# ============================================================
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_root)
        self.transform = transform or DEFAULT_TRANSFORM
        self.image_map = {}
        for sub in sorted(self.img_dir.glob("images_*")):
            nested = sub / "images"
            if nested.exists():
                for f in nested.glob("*.png"):
                    self.image_map[f.name] = f
        labels = self.df["Finding Labels"].str.split("|").explode().unique()
        self.classes = sorted([l for l in labels if l != "No Finding"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["Image Index"]
        img_path = self.image_map.get(fname)
        if img_path is None or not img_path.exists():
            dummy = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            target = torch.zeros(len(self.classes))
            return dummy, target
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        labels = row["Finding Labels"].split("|")
        target = torch.zeros(len(self.classes))
        for l in labels:
            if l in self.class_to_idx:
                target[self.class_to_idx[l]] = 1.0
        return img, target

# ============================================================
# HAM10000 Skin Dataset
# ============================================================
class SkinDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        self.meta = pd.read_csv(root / "HAM10000_metadata.csv")
        self.root = root
        self.transform = transform or DEFAULT_TRANSFORM
        self.classes = sorted(self.meta["dx"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.root / "ham10000_images_part_1" / f"{row['image_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        y = torch.tensor(self.class_to_idx[row["dx"]], dtype=torch.long)
        return img, y

# ============================================================
# Brain MRI Dataset
# ============================================================
class MRIDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        valid_exts = [".jpg", ".jpeg", ".png"]

        # Filter only real image-containing subfolders
        subdirs = [
            d for d in root.iterdir()
            if d.is_dir() and any(f.suffix.lower() in valid_exts for f in d.iterdir())
        ]

        self.samples = []
        for sub in sorted(subdirs):
            for f in sub.rglob("*"):
                if f.suffix.lower() in valid_exts:
                    self.samples.append((f, sub.name))

        self.transform = transform or DEFAULT_TRANSFORM
        self.classes = sorted(list({lab for _, lab in self.samples}))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        label_idx = self.class_to_idx[label]
        assert 0 <= label_idx < len(self.classes), f"Invalid label {label} -> {label_idx}"
        return img, torch.tensor(label_idx, dtype=torch.long)

# ============================================================
# DataLoader Getter
# ============================================================
def get_dataloader(modality, batch_size=8, shuffle=True, num_workers=2, transform=None):
    m = modality.upper()
    if m == "XRAY":
        csv = KAGGLE_PATHS["XRAY"] / "Data_Entry_2017.csv"
        imgs_root = KAGGLE_PATHS["XRAY"]
        ds = NIHChestXrayDataset(csv, imgs_root, transform)
    elif m == "SKIN":
        ds = SkinDataset(KAGGLE_PATHS["SKIN"], transform)
    elif m == "MRI":
        ds = MRIDataset(KAGGLE_PATHS["MRI"], transform)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
