import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from numpy import array
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    multilabel_confusion_matrix
)

from config import DEVICE, CHECKPOINT_DIR, BATCH_SIZE
from data.dataset_loader import get_dataloader
from models.multitask_vit import MultiTaskViT


def fix_state_dict_keys(state_dict):
    """Removes 'module.' prefix from keys if present."""
    if next(iter(state_dict.keys())).startswith("module."):
        print("🧩 Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model(ckpt_path, n_xray, n_skin, n_mri):
    model = MultiTaskViT(n_xray, n_skin, n_mri)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Fix state dict keys (e.g., remove 'module.' prefix)
    state_dict = fix_state_dict_keys(ckpt["model_state"])
    
    # Load the corrected state dict
    model.load_state_dict(state_dict, strict=True) 
    
    # Add DataParallel if multiple GPUs are used for evaluation
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs for evaluation.")
        model = nn.DataParallel(model)
        
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded model from {ckpt_path}")
    return model


@torch.no_grad()
def evaluate(task, model, dataloader, class_names=None):
    preds, trues = [], []
    pbar = tqdm(dataloader, desc=f"Evaluating {task}")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs, task)

        if task == "XRAY":
            # Multi-label classification
            probs = torch.sigmoid(outputs)
            preds.extend((probs > 0.5).int().cpu().numpy())
            trues.extend(labels.cpu().numpy())
        else:
            # Multi-class (or binary) classification
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    preds, trues = array(preds), array(trues)

    # --- Metrics Calculation ---
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    prec = precision_score(trues, preds, average="macro", zero_division=0)
    rec = recall_score(trues, preds, average="macro", zero_division=0)

    print(f"✅ {task} → ACC={acc:.4f}, F1={f1:.4f}, PREC={prec:.4f}, REC={rec:.4f}")

    # --- Confusion Matrix Generation ---
    if task == "XRAY":
        print(f"\n--- Multilabel Confusion Matrix for {task} ---")
        if class_names:
            print(f"Classes: {class_names}")
        print("(One 2x2 [TN, FP, FN, TP] matrix per class)")
        mcm = multilabel_confusion_matrix(trues, preds)
        print(mcm)
    else:
        print(f"\n--- Confusion Matrix for {task} ---")
        if class_names:
            print(f"Classes: {class_names}")
        cm = confusion_matrix(trues, preds)
        print(cm)
        
    return acc, f1, prec, rec


def main():
    print("Device:", DEVICE)
    
    # --- Load Data for Evaluation ---
    # We use shuffle=False to ensure a consistent evaluation set.
    print("Loading evaluation datasets (with shuffle=False)...")
    xray_dl = get_dataloader("XRAY", BATCH_SIZE, shuffle=False)
    skin_dl = get_dataloader("SKIN", BATCH_SIZE, shuffle=False)
    mri_dl  = get_dataloader("MRI", BATCH_SIZE, shuffle=False)

    # --- Get Class Info ---
    nx = len(xray_dl.dataset.classes)
    ns = len(skin_dl.dataset.classes)
    nm = len(mri_dl.dataset.classes)
    
    # --- Load Model ---
    ckpt_path = "/kaggle/input/vit-21/pytorch/default/1/vit_epoch21_skin_mri_only.pt"  # change this if needed
    model = load_model(ckpt_path, nx, ns, nm)

    # --- Run Evaluation ---
    # Pass class_names to the evaluate function for labeled matrices
    
    # Note: Your loaded checkpoint is 'skin_mri_only.pt'
    # Evaluating XRAY might give poor results if it wasn't trained.
    print("\n--- Evaluating XRAY Task ---")
    evaluate("XRAY", model, xray_dl, class_names=xray_dl.dataset.classes)
    
    print("\n--- Evaluating SKIN Task ---")
    evaluate("SKIN", model, skin_dl, class_names=skin_dl.dataset.classes)
    
    print("\n--- Evaluating MRI Task ---")
    evaluate("MRI", model, mri_dl, class_names=mri_dl.dataset.classes)


if __name__ == "__main__":
    main()
