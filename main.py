# main.py
"""
Simple orchestrator for loading data, running model inference and explainability.
This script demonstrates the multimodal routing for 2D (LIME) and 3D (Grad-CAM).
"""

import numpy as np
from PIL import Image
import torch
from data.dataset_loader import get_dataloader
from models.vit_model import MultimodalViTWrapper
from explainability.lime_explainer import LIME2DExplainer
from explainability.gradcam_explainer import GradCAM3DExplainer # <-- Updated
from utils.visualization import show_image_with_mask, show_3d_explanation, show_metrics
from utils.metrics import evaluate_model
from config import MODALITY_CONFIG, DEVICE

def preprocess_2d_for_pil(batch_images):
    # batch_images: list of 2D image tensors (C, H, W)
    pil_imgs = []
    for img_tensor in batch_images:
        arr = img_tensor.permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).astype('uint8')
        
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
            
        pil_imgs.append(Image.fromarray(arr))
    return pil_imgs

def demo_modality(modality_type, dataset_name, multimodal_model, sample_count=4):
    print(f"\n========================================================")
    print(f"DEMO: Running inference and explanation for {modality_type}")
    print(f"========================================================")

    # 1. Load Data
    dl = get_dataloader(dataset_name, modality=modality_type, batch_size=sample_count, shuffle=False)
    
    # Grab one batch
    # Note: MONAI dataloader returns a dict
    batch = next(iter(dl))

    # 2. Prepare Inputs based on modality
    if modality_type in ["XRAY", "ULTRASOUND"]:
        input_tensors, labels = batch[0], batch[1]
        pil_imgs = preprocess_2d_for_pil(input_tensors)
        model_input = pil_imgs
        explainer_input = np.array(pil_imgs[0]) # First sample for LIME
        
    elif modality_type in ["CT", "MRI"]:
        input_tensors, labels = batch["image"], batch["label"]
        model_input = input_tensors # List of (C, D, H, W) tensors
        explainer_input = model_input[0].unsqueeze(0).to(DEVICE) # (1, C, D, H, W)
        
    else:
        raise ValueError("Modality not supported in demo.")
        
    # 3. Prediction and Evaluation
    probs = multimodal_model.predict(model_input, modality_type)
    preds = probs.argmax(axis=1)
    
    # Ensure labels are flat numpy array for evaluation
    true_labels_np = labels.cpu().numpy().flatten()[:sample_count]
    report = evaluate_model(true_labels_np, preds, labels=[0, 1])
    show_metrics(report, title=f"Classification Report for {modality_type}")
    
    # 4. Explainability (Routing)
    print(f"\n--- Explanation for first sample (pred: {preds[0]}) ---")

    if modality_type in ["XRAY", "ULTRASOUND"]:
        # 2D Explainer (LIME)
        def predict_wrapper_2d(images_np):
            pil_list = [Image.fromarray(x.astype('uint8')) for x in images_np]
            return multimodal_model.predict(pil_list, modality_type)
            
        lime = LIME2DExplainer(predict_wrapper_2d)
        explanation, temp, mask = lime.explain(explainer_input, label=preds[0], num_features=10)
        show_image_with_mask(explainer_input, mask, title=f"LIME 2D Mask for {modality_type} (pred {preds[0]})")
        
    elif modality_type in ["CT", "MRI"]:
        # 3D Explainer (Grad-CAM)
        raw_3d_model = multimodal_model.get_model(modality_type)
        
        # Target layer for MONAI ViT
        target_layer = "model.transformer.blocks[-1].norm1" 
        
        gradcam = GradCAM3DExplainer(raw_3d_model, target_layer)
        heatmap = gradcam.explain(explainer_input, label=preds[0])
        
        # Show explanation
        show_3d_explanation(
            explainer_input.squeeze(0), # (C, D, H, W)
            heatmap[0], # (1, D, H, W)
            title=f"3D Grad-CAM for {modality_type} (pred {preds[0]})"
        )


def main_demo():
    print("Initializing Multimodal Model Router...")
    multimodal_vit = MultimodalViTWrapper()

    # Run 2D Demo (XRAY)
    demo_modality(
        modality_type="XRAY", 
        dataset_name="chestmnist", 
        multimodal_model=multimodal_vit, 
        sample_count=4
    )
    
    # Run 3D Demo (CT)
    # This will use the dummy NIfTI files created by the data loader
    demo_modality(
        modality_type="CT", 
        dataset_name="none", # Not used, path is built from config
        multimodal_model=multimodal_vit, 
        sample_count=2
    )

if __name__ == "__main__":
    main_demo()
