# main.py
"""
Inference and Explanation script.

This script loads the model trained by 'train_vit_multi_task.py' 
to perform classification and generate 2D explanations:
- LIME
- SHAP
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os
from pathlib import Path

# Import the correct model and data loader
from data.dataset_loader import get_dataloader
from models.multitask_vit import MultiTaskViT 
from explainability.lime_explainer import LIME2DExplainer
from explainability.shap_explainer import SHAP2DExplainer
from utils.visualization import (
    show_image_with_mask, show_shap_explanation, show_metrics
)
from utils.metrics import evaluate_model
from config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR, 
    SHAP_NSAMPLES, VAL_TRANSFORM
)

def load_trained_model(num_xray, num_skin, num_mri):
    """
    Loads the latest checkpoint from the CHECKPOINT_DIR.
    """
    model = MultiTaskViT(num_xray, num_skin, num_mri)
    
    # Load checkpoint
    ckpts = sorted(Path(CHECKPOINT_DIR).glob("vit_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}. Did you run train_vit_multi_task.py?")
        
    latest_ckpt_path = ckpts[-1]
    print(f"Loading checkpoint: {latest_ckpt_path}")
    
    # Load state dict
    ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
    
    # Adjust for DataParallel prefix 'module.' if it was saved with it
    model_state = ckpt["model_state"]
    if next(iter(model_state.keys())).startswith("module."):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        
    model.load_state_dict(model_state)
    model = model.to(DEVICE)
    model.eval()
    return model

def get_denormalizer():
    """Returns a function to denormalize images for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def denormalize(tensor):
        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)
        return transforms.ToPILImage()(tensor)
        
    return denormalize

def predict_fn_wrapper(model, task):
    """
    Wraps the model's forward pass for the explainers.
    Handles numpy -> tensor conversion.
    """
    def predict_proba(numpy_images):
        # Convert (N, H, W, C) numpy array to (N, C, H, W) tensor
        pil_images = [Image.fromarray(img.astype('uint8')) for img in numpy_images]
        tensors = torch.stack([VAL_TRANSFORM(img) for img in pil_images]).to(DEVICE)
        
        with torch.no_grad():
            logits = model(tensors, task)
            
            # Use Softmax for single-label (SKIN, MRI)
            if task in ["SKIN", "MRI"]:
                probs = torch.softmax(logits, dim=1)
            # Use Sigmoid for multi-label (XRAY)
            else:
                probs = torch.sigmoid(logits)
                
        return probs.cpu().numpy()
        
    return predict_proba


def demo_modality_2d(model, task_name, class_names, sample_count=4):
    """
    Runs prediction and explanation for a 2D modality (SKIN or MRI).
    """
    print(f"\n========================================================")
    print(f"DEMO: {task_name}")
    print(f"========================================================")

    # 1. Load Data (using shuffle=False to get consistent results)
    try:
        # Use validation transform, no shuffling
        dl = get_dataloader(task_name, batch_size=sample_count, shuffle=False)
        batch = next(iter(dl))
        input_tensors, labels = batch[0], batch[1]
    except Exception as e:
        print(f"Could not load data for {task_name}: {e}")
        return

    # 2. Prepare Inputs
    input_tensors = input_tensors.to(DEVICE)
    labels = labels.to(DEVICE)
    denormalize = get_denormalizer()
    
    # Get numpy images for visualization
    vis_input_list = [np.array(denormalize(t)) for t in input_tensors.cpu()]
    
    # We'll explain the first image
    explainer_input_np = vis_input_list[0]
    # We'll use the rest of the batch as the SHAP background
    shap_background_np = np.stack(vis_input_list[1:])

    # 3. Prediction and Evaluation
    with torch.no_grad():
        logits = model(input_tensors, task_name)
    
    preds = logits.argmax(axis=1).cpu().numpy()
    true_labels_np = labels.cpu().numpy().flatten()
    
    report = evaluate_model(true_labels_np, preds, labels=list(range(len(class_names))))
    show_metrics(report, title=f"Batch Classification Report for {task_name}")

    # 4. Explainability
    pred_index = preds[0]
    pred_name = class_names[pred_index]
    
    print(f"\n--- Explanation for first sample (Pred: {pred_index} - '{pred_name}') ---")

    # Create the prediction function for explainers
    predict_proba = predict_fn_wrapper(model, task_name)

    # --- LIME Explanation ---
    print("Running LIME...")
    lime = LIME2DExplainer(predict_proba)
    explanation, temp, mask = lime.explain(explainer_input_np, label=pred_index, num_features=10)
    
    lime_save_path = RESULTS_DIR / f"lime_explanation_{task_name}.png"
    show_image_with_mask(explainer_input_np, mask, 
                         title=f"LIME 2D Mask for {task_name} (Pred: '{pred_name}')",
                         save_path=lime_save_path)
    print(f"LIME explanation saved to {lime_save_path}")

    # --- SHAP Explanation ---
    print("Running SHAP... (This may take a moment)")
    shap_explainer = SHAP2DExplainer(predict_proba, shap_background_np)
    shap_values = shap_explainer.explain(
        explainer_input_np.reshape(1, *explainer_input_np.shape), 
        nsamples=SHAP_NSAMPLES
    )
    
    shap_save_path = RESULTS_DIR / f"shap_explanation_{task_name}.png"
    show_shap_explanation(
        shap_values, 
        explainer_input_np,
        class_names=class_names,
        title=f"SHAP Explanation for {task_name} (Pred: '{pred_name}')",
        save_path=shap_save_path
    )
    print(f"SHAP explanation saved to {shap_save_path}")


def main_demo():
    print("--- Running Inference & Explanation ---")
    
    # 1. Get class maps from data loader
    # We load a temporary dataloader just to get class info
    try:
        print("Loading class information...")
        xray_dl = get_dataloader("XRAY", BATCH_SIZE)
        skin_dl = get_dataloader("SKIN", BATCH_SIZE)
        mri_dl  = get_dataloader("MRI", BATCH_SIZE)

        class_names_xray = xray_dl.dataset.classes
        class_names_skin = skin_dl.dataset.classes
        class_names_mri = mri_dl.dataset.classes
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure your datasets are correctly configured in config.py")
        return

    # 2. Load the trained model
    try:
        model = load_trained_model(
            num_xray=len(class_names_xray),
            num_skin=len(class_names_skin),
            num_mri=len(class_names_mri)
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Run Demos (2D tasks are best for LIME/SHAP)
    demo_modality_2d(
        model=model,
        task_name="SKIN", 
        class_names=class_names_skin,
        sample_count=4 # 1 to explain, 3 for background
    )
    
    demo_modality_2d(
        model=model,
        task_name="MRI", 
        class_names=class_names_mri,
        sample_count=4 # 1 to explain, 3 for background
    )

if __name__ == "__main__":
    main_demo()
