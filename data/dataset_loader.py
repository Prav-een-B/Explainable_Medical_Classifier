# data/dataset_loader.py
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        self.root = root
        self.meta = pd.read_csv(root / "HAM10000_metadata.csv")
        self.transform = transform or TRANSFORM
        self.classes = sorted(self.meta["dx"].unique())
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.root / "ham10000_images_part_1" / f"{row['image_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[row["dx"]]
        return img, torch.tensor(label, dtype=torch.long)

class MRISliceDataset(Dataset):
    """
    Safe MRI dataset that treats each sample as an image (if provided as jpg/png).
    If your MRI dataset is not slice images, convert/preprocess separately.
    """
    def __init__(self, root, transform=None):
        root = Path(root)
        self.samples = []
        for p in root.rglob("*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                # label by parent folder name
                label = p.parent.name
                self.samples.append((p, label))
        self.transform = transform or TRANSFORM
        self.classes = sorted(list({lab for _,lab in self.samples}))
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, lab = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.class_to_idx[lab], dtype=torch.long)

def get_dataloader(modality, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    m = modality.upper()
    if m == "XRAY":
        csv = KAGGLE_PATHS["XRAY"] / "Data_Entry_2017.csv"
        ds = NIHChestXrayDataset(csv, KAGGLE_PATHS["XRAY"])
    elif m == "SKIN":
        ds = SkinDataset(KAGGLE_PATHS["SKIN"])
    elif m == "MRI":
        ds = MRISliceDataset(KAGGLE_PATHS["MRI"])
    else:
        raise ValueError("Unknown modality")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
