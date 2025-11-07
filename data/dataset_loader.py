# data/dataset_loader.py
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import VAL_TRANSFORM, IMG_SIZE_2D
from config import KAGGLE_PATHS, IMG_SIZE_2D, BATCH_SIZE, NUM_WORKERS

# Single transform used for all 2D tasks
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE_2D, IMG_SIZE_2D)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, skip_missing=True):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root_dir)
        self.transform = transform or TRANSFORM
        self.skip_missing = skip_missing

        # build a filename -> path map from nested images_* directories
        self.image_map = {}
        for p in sorted(self.root.glob("images_*")):
            imgdir = p / "images"
            if imgdir.exists():
                for f in imgdir.glob("*"):
                    self.image_map[f.name] = f

        # class list (exclude 'No Finding' from class list or keep depending on use)
        all_labels = self.df["Finding Labels"].str.split("|").explode().unique()
        classes = [c for c in sorted(all_labels) if str(c) and c != "No Finding"]
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["Image Index"]
        p = self.image_map.get(fname)
        if p is None or not p.exists():
            if self.skip_missing:
                img = Image.new("RGB", (IMG_SIZE_2D, IMG_SIZE_2D))
            else:
                raise FileNotFoundError(f"{fname} not found under {self.root}")
        else:
            img = Image.open(p).convert("RGB")
        img = self.transform(img)
        labels = row["Finding Labels"].split("|")
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for lab in labels:
            if lab in self.class_to_idx:
                target[self.class_to_idx[lab]] = 1.0
        return img, target

class SkinDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        self.meta = pd.read_csv(root / "HAM10000_metadata.csv")
        self.root = root
        self.transform = transform 
        self.classes = sorted(self.meta["dx"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # --- FIX: Handle uppercase/lowercase folders ---
        self.image_map = {}
        possible_folders = [
            "HAM10000_images_part_1", "HAM10000_images_part_2",
            "ham10000_images_part_1", "ham10000_images_part_2"
        ]
        for folder_name in possible_folders:
            folder = root / folder_name
            if folder.exists():
                for f in folder.glob("*.jpg"):
                    self.image_map[f.stem] = f

        if not self.image_map:
            raise FileNotFoundError("No image folders found under HAM10000 dataset root")

        print(f"✅ Found {len(self.image_map)} total images in HAM10000 dataset.")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_id = row["image_id"]
        img_path = self.image_map.get(img_id)

        # --- FIX: Skip missing files gracefully ---
        if img_path is None or not img_path.exists():
            dummy = torch.zeros(3, IMG_SIZE_2D, IMG_SIZE_2D, dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return dummy, y

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y = torch.tensor(self.class_to_idx[row["dx"]], dtype=torch.long)
        return img, y


class MRIDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        self.transform = transform or VAL_TRANSFORM
        self.root = root

        # --- FIX: Support different folder naming conventions ---
        possible_folders = ["yes", "no", "Yes", "No", "Tumor", "NoTumor"]
        self.image_paths = []
        self.labels = []

        for folder_name in possible_folders:
            folder = root / folder_name
            if folder.exists():
                label = 1 if "yes" in folder_name.lower() or "tumor" in folder_name.lower() else 0
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img_path in folder.glob(ext):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        if not self.image_paths:
            raise FileNotFoundError(
                f"❌ No valid MRI image folders found under {root}. "
                "Expected 'yes/no' or 'Tumor/NoTumor' structure."
            )

        self.classes = ["NoTumor", "Tumor"]
        self.class_to_idx = {"NoTumor": 0, "Tumor": 1}

        print(f"✅ Found {len(self.image_paths)} MRI images in total under {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # --- FIX: Handle missing or corrupt files gracefully ---
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping unreadable file: {img_path} ({e})")
            dummy = torch.zeros(3, IMG_SIZE_2D, IMG_SIZE_2D, dtype=torch.float32)
            return dummy, torch.tensor(0, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(label, dtype=torch.long)
        return img, y


def get_dataloader(modality, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    m = modality.upper()
    if m == "XRAY":
        csv = KAGGLE_PATHS["XRAY"] / "Data_Entry_2017.csv"
        ds = NIHChestXrayDataset(csv, KAGGLE_PATHS["XRAY"])
    elif m == "SKIN":
        ds = SkinDataset(KAGGLE_PATHS["SKIN"])
    elif m == "MRI":
        ds = MRIDataset(KAGGLE_PATHS["MRI"])
    else:
        raise ValueError("Unknown modality")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
