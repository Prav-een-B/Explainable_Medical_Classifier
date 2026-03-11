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


# ============================
# ✅ Utility: Safe key-fix function
# ============================
def fix_state_dict_keys(state_dict, model_state):
    has_module_ckpt = any(k.startswith("module.") for k in state_dict.keys())
    has_module_model = any(k.startswith("module.") for k in model_state.keys())
    if has_module_model and not has_module_ckpt:
        print("🧩 Adding 'module.' prefix (single-GPU → multi-GPU)")
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not has_module_model and has_module_ckpt:
        print("🧩 Removing 'module.' prefix (multi-GPU → single-GPU)")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ============================
# ✅ Training Function
# ============================
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
            loss = crit(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        count += 1
        pbar.set_postfix_str(f"loss={total/count:.4f}")
    return total / max(1, count)


# ============================
# ✅ Model + Optimizer Creation
# ============================
def create_model_and_optim(n_xray, n_skin, n_mri):
    model = MultiTaskViT(n_xray, n_skin, n_mri)

    # Wrap model first (so DataParallel keys are consistent)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # Load checkpoint if exists
    ckpts = sorted(Path(CHECKPOINT_DIR).glob("vit_epoch*.pt"))
    if ckpts:
        latest = ckpts[-1]
        print("Resuming from", latest)
        ckpt = torch.load(latest, map_location=DEVICE)
        state_dict = fix_state_dict_keys(ckpt["model_state"], model.state_dict())
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint. Missing={len(missing)}, Unexpected={len(unexpected)}")

    opt = optim.AdamW(model.parameters(), lr=LR)
    return model, opt


# ============================
# ✅ Checkpoint Saver
# ============================
def save_checkpoint(epoch, model, opt, scaler, suffix, out_dir=CHECKPOINT_DIR):
    ckpt = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "opt_state": opt.state_dict(),
        "scaler_state": scaler.state_dict()
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"vit_epoch{epoch:02d}_skin_mri_only.pt"
    torch.save(ckpt, path)
    print(f"💾 Saved checkpoint: {path}")


# ============================
# ✅ Main Training Loop
# ============================
def main():
    print("Device:", DEVICE)
    xray_dl = get_dataloader("XRAY", BATCH_SIZE)
    skin_dl = get_dataloader("SKIN", BATCH_SIZE)
    mri_dl  = get_dataloader("MRI", BATCH_SIZE)

    nx = len(xray_dl.dataset.classes)
    ns = len(skin_dl.dataset.classes)
    nm = len(mri_dl.dataset.classes)

    model, opt = create_model_and_optim(nx, ns, nm)

    crit_x = nn.BCEWithLogitsLoss()
    crit_ce = nn.CrossEntropyLoss()
    scaler = amp.GradScaler() if USE_AMP else None

    # --- Resume optimizer/scaler only ---
    ckpts = sorted(Path(CHECKPOINT_DIR).glob("vit_epoch*.pt"))
    start_epoch = 9
    if ckpts:
        latest = ckpts[-1]
        print("🔁 Resuming optimizer/scaler from", latest)
        ckpt = torch.load(latest, map_location=DEVICE)

        # ✅ FIX KEY PREFIX HERE ALSO BEFORE LOADING MODEL STATE (for safety)
        state_dict = fix_state_dict_keys(ckpt["model_state"], model.state_dict())
        model.load_state_dict(state_dict, strict=False)

        try:
            opt.load_state_dict(ckpt["opt_state"])
        except Exception as e:
            print("⚠️ Optimizer load skipped:", e)
        if scaler and "scaler_state" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print("⚠️ Scaler load skipped:", e)

        start_epoch = ckpt.get("epoch", 4) + 1
        print(f"✅ Resumed up to epoch {start_epoch-1}. Continuing from {start_epoch}.")
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")

    end_epoch = start_epoch + 3
    print(f"Training epochs {start_epoch} → {end_epoch}")

    for e in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        ls = train_one("SKIN", skin_dl, model, opt, crit_ce, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "skin_done")

        lm = train_one("MRI", mri_dl, model, opt, crit_ce, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "mri_done")

        # lx = train_one("XRAY", xray_dl, model, opt, crit_x, scaler if scaler else amp.GradScaler(enabled=False))
        # save_checkpoint(e, model, opt, scaler, "xray_final")
        lx = 0.0

        print(f"✅ Epoch {e} complete | XRAY={lx:.4f}, SKIN={ls:.4f}, MRI={lm:.4f} | Time={(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
