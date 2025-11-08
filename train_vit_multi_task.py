# train_vit_multi_task.py
import time, os
from pathlib import Path
import torch
import torch.nn as nn, torch.optim as optim
from torch import amp
from tqdm import tqdm

from config import DEVICE, EPOCHS, LR, BATCH_SIZE, CHECKPOINT_DIR, USE_AMP
from data.dataset_loader import get_dataloader
from models.multitask_vit import MultiTaskViT

def train_one(task, dl, model, opt, crit, scaler):
    model.train()
    total = 0.0
    count = 0
    pbar = tqdm(dl, desc=f"Train {task}", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        opt.zero_grad()
        with amp.autocast(device_type="cuda", enabled=USE_AMP):
            preds = model(imgs, task)
            # choose loss type: XRAY -> BCEWithLogits (multi-label), others -> CrossEntropy
            loss = crit(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        count += 1
        pbar.set_postfix_str(f"loss={total/count:.4f}")
    return total / max(1, count)

def create_model_and_optim(n_xray, n_skin, n_mri):
    model = MultiTaskViT(n_xray, n_skin, n_mri)

    # --- Load checkpoint before wrapping ---
    ckpts = sorted(Path(CHECKPOINT_DIR).glob("vit_epoch*.pt"))
    if ckpts:
        latest = ckpts[-1]
        print("Resuming from", latest)
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print("✅ Loaded checkpoint successfully")

    # Now wrap if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR)
    return model, opt

# --- MODIFIED: Added 'suffix' argument to save intermediate checkpoints ---
def save_checkpoint(epoch, model, opt, scaler, suffix, out_dir=CHECKPOINT_DIR):
    ckpt = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "opt_state": opt.state_dict(),
        "scaler_state": scaler.state_dict()
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # --- MODIFIED: Filename now includes the suffix ---
    path = Path(out_dir) / f"vit_epoch{epoch:02d}_{suffix}.pt"
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")

def main():
    print("Device:", DEVICE)
    xray_dl = get_dataloader("XRAY", BATCH_SIZE)
    skin_dl = get_dataloader("SKIN", BATCH_SIZE)
    mri_dl  = get_dataloader("MRI", BATCH_SIZE)

    nx = len(xray_dl.dataset.classes)
    ns = len(skin_dl.dataset.classes)
    nm = len(mri_dl.dataset.classes)

    model, opt = create_model_and_optim(nx, ns, nm)

    # losses
    crit_x = nn.BCEWithLogitsLoss()
    crit_ce = nn.CrossEntropyLoss()

    scaler = amp.GradScaler() if USE_AMP else None

    start_epoch = 1
    # resume pick the latest checkpoint automatically (if exists)
    ckpts = sorted(Path(CHECKPOINT_DIR).glob("vit_epoch*.pt"))
    if ckpts:
        latest = ckpts[-1]
        print("Resuming from", latest)
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        if scaler and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        
        # --- MODIFIED: Robust resume logic ---
        # Get the epoch from the checkpoint. We will re-run this epoch
        # to ensure all tasks are completed, even if interrupted.
        start_epoch = ckpt.get("epoch", 1)
        print(f"Restarting at epoch {start_epoch} to ensure all tasks are complete.")

    for e in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        # --- MODIFIED: Save checkpoint after each task ---
        
        # 1. TRAIN SKIN
        ls = train_one("SKIN", skin_dl, model, opt, crit_ce, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "skin_done")

        # 2. TRAIN MRI
        lm = train_one("MRI", mri_dl, model, opt, crit_ce, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "mri_done")

        # 3. TRAIN XRAY (Last, as requested)
        lx = train_one("XRAY", xray_dl, model, opt, crit_x, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "xray_final") # This is the final save for the epoch
        
        print(f"Epoch {e}: XRAY={lx:.4f}, SKIN={ls:.4f}, MRI={lm:.4f}  time={(time.time()-t0)/60:.2f}min")

if __name__=="__main__":
    main()
