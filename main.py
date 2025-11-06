# main.py
"""
Inference and Explanation script.

This script loads the models trained by 'train.py' to perform classification
and generate explanations:
- LIME (for 2D)
- SHAP (for 2D)
- Grad-CAM (for 3D)
"""

import numpy as np
from PIL import Image
import torch
import json
import os
from torchvision import transforms # Added for denormalizing

# Import data loader *only* for helper functions and class map
from data.dataset_loader import get_dataloader, load_class_map, get_2d_transforms
from models.vit_model import MultimodalViTWrapper
from explainability.lime_explainer import LIME2DExplainer
from explainability.gradcam_explainer import GradCAM3DExplainer
from explainability.shap_explainer import SHAP2DExplainer # ADDED
from utils.visualization import (
    show_image_with_mask, show_3d_explanation, 
    show_metrics, show_shap_explanation
)
from utils.metrics import evaluate_model
from config import MODALITY_CONFIG, DEVICE, IMG_SIZE_2D, RESULTS_DIR, SHAP_NSAMPLES

def get_denormalized_images(batch_tensors):
    """
    Denormalizes a batch of 2D tensors for visualization.
    """
    vis_input_list = []
    pil_input_list = []
    
    # ViT normalization is mean=0.5, std=0.5
    # Denorm: (tensor * 0.5) + 0.5
    
    for img_tensor in batch_tensors:
        # Denormalize from [-1, 1] to [0, 1]
        denorm_tensor = (img_tensor * 0.5) + 0.5
        denorm_tensor = denorm_tensor.clamp(0, 1)
        
        # Create PIL image for model (expects 0-255)
        pil_img = transforms.ToPILImage()(denorm_tensor)
        pil_input_list.append(pil_img)
        
        # Create numpy array for explainers (expects 0-255)
        vis_input_list.append(np.array(pil_img))
            
    return pil_input_list, vis_input_list

def demo_modality(modality_type, multimodal_model, class_map, class_names_list, sample_count=4):
    print(f"\n========================================================")
    print(f"DEMO: {modality_type} (Dataset: {modality_type})")
    print(f"========================================================")

    # 1. Load Data (using 'test' split)
    try:
        dl = get_dataloader(modality_type, class_map, split="test", batch_size=sample_count, shuffle=False)
        batch = next(iter(dl))
    except Exception as e:
        print(f"Could not load data for {modality_type}: {e}")
        return

    # 2. Prepare Inputs based on modality
    if modality_type in ["XRAY", "HISTOPATHOLOGY"]:
        input_tensors, labels = batch[0], batch[1]
        
        model_input, vis_input_list = get_denormalized_images(input_tensors)
        
        # We'll explain the first image
        explainer_input_np = vis_input_list[0] 
        # We'll use the rest of the batch as the SHAP background
        shap_background_np = np.stack(vis_input_list[1:])
        
    elif modality_type == "MRI":
        input_tensors, labels = batch["image"], batch["label"]
        model_input = input_tensors 
        explainer_input_tensor = model_input[0].unsqueeze(0).to(DEVICE)
        
    else:
        raise ValueError("Modality not supported in demo.")
        
    # 3. Prediction and Evaluation
    probs = multimodal_model.predict(model_input, modality_type)
    preds = probs.argmax(axis=1)
    
    true_labels_np = labels.cpu().numpy().flatten()[:sample_count]
    report = evaluate_model(true_labels_np, preds, labels=list(range(len(class_names_list))))
    show_metrics(report, title=f"Classification Report for {modality_type}")
    
    # Save the full report to results/
    report_filename = RESULTS_DIR / f"report_{modality_type}.txt"
    with open(report_filename, 'w') as f:
        f.write(f"Classification Report for {modality_type}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_filename}")
    
    # 4. Explainability (Routing)
    pred_index = preds[0]
    pred_name = class_names_list[pred_index]
    
    print(f"\n--- Explanation for first sample (Pred: {pred_index} - '{pred_name}') ---")

    if modality_type in ["XRAY", "HISTOPATHOLOGY"]:
        
        # --- LIME Explanation ---
        print("Running LIME...")
        def predict_wrapper_2d(images_np_list):
            pil_list = [Image.fromarray(x.astype('uint8')) for x in images_np_list]
            return multimodal_model.predict(pil_list, modality_type)
            
        lime = LIME2DExplainer(predict_wrapper_2d)
        explanation, temp, mask = lime.explain(explainer_input_np, label=pred_index, num_features=10)
        
        lime_save_path = RESULTS_DIR / f"lime_explanation_{modality_type}.png"
        show_image_with_mask(explainer_input_np, mask, 
                             title=f"LIME 2D Mask for {modality_type} (Pred: '{pred_name}')",
                             save_path=lime_save_path)
        print(f"LIME explanation saved to {lime_save_path}")

        # --- SHAP Explanation (Point 4) ---
        print("Running SHAP... (This may take a moment)")
        # We pass the same predict_wrapper
        shap_explainer = SHAP2DExplainer(predict_wrapper_2d, shap_background_np)
        
        # Explain a single image
        shap_values = shap_explainer.explain(
            explainer_input_np.reshape(1, *explainer_input_np.shape), 
            nsamples=SHAP_NSAMPLES
        )
        
        shap_save_path = RESULTS_DIR / f"shap_explanation_{modality_type}.png"
        show_shap_explanation(
            shap_values, 
            explainer_input_np,
            class_names=class_names_list,
            title=f"SHAP Explanation for {modality_type} (Pred: '{pred_name}')",
            save_path=shap_save_path
        )
        print(f"SHAP explanation saved to {shap_save_path}")

        
    elif modality_type == "MRI":
        # --- 3D Grad-CAM Explanation ---
        print("Running 3D Grad-CAM...")
        raw_3d_model = multimodal_model.get_model(modality_type)
        target_layer = "model.transformer.blocks[-1].norm1" 
        
        gradcam = GradCAM3DExplainer(raw_3d_model, target_layer)
        heatmap = gradcam.explain(explainer_input_tensor, label=pred_index)
        
        gradcam_save_path = RESULTS_DIR / f"gradcam_explanation_{modality_type}.png"
        show_3d_explanation(
            explainer_input_tensor.squeeze(0), 
            heatmap[0],
            title=f"3D Grad-CAM for {modality_type} (Pred: '{pred_name}')",
            save_path=gradcam_save_path
        )
        print(f"Grad-CAM explanation saved to {gradcam_save_path}")


def main_demo():
    print("--- Running Inference & Explanation ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load the dynamic class map created by train.py
    try:
        class_map = load_class_map()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    num_labels = len(class_map)
    class_names_list = {v: k for k, v in class_map.items()}

    # 2. Initialize Wrapper in INFERENCE mode
    try:
        multimodal_vit = MultimodalViTWrapper(num_labels=num_labels, load_from_scratch=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Could not load trained models. Did you run train.py successfully?")
        return

    # Run 2D Demo (XRAY)
    demo_modality(
        modality_type="XRAY", 
        multimodal_model=multimodal_vit,
        class_map=class_map,
        class_names_list=class_names_list,
        sample_count=4 # 1 to explain, 3 for background
    )
    
    # Run 2D Demo (HISTOPATHOLOGY)
    demo_modality(
        modality_type="HISTOPATHOLOGY", 
        multimodal_model=multimodal_vit,
        class_map=class_map,
        class_names_list=class_names_list,
        sample_count=4 # 1 to explain, 3 for background
    )
    
    # Run 3D Demo (MRI)
    demo_modality(
        modality_type="MRI", 
        multimodal_model=multimodal_vit,
        class_map=class_map,
        class_names_list=class_names_list,
        sample_count=2
    )

if __name__ == "__main__":
    main_demo()
