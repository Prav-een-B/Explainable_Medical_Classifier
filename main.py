# main.py
"""
Simple orchestrator for loading data, running model inference and explainability.
This script demonstrates the multimodal routing and LIME-based explanation on samples.
"""

import numpy as np
from PIL import Image
import torch
from data.dataset_loader import get_dataloader
from models.vit_model import MultimodalViTWrapper
from explainability.lime_explainer import LIME2DExplainer, LIME3DExplainer
from utils.visualization import show_image_with_mask, show_3d_explanation, show_metrics
from utils.metrics import evaluate_model
from config import MODALITY_CONFIG

def preprocess_2d_for_pil(batch_images):
    # batch_images: list of 2D image tensors (C, H, W)
    pil_imgs = []
    for img in batch_images:
        arr = img.permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).astype('uint8')
        
        # Handle 1-channel images by converting to RGB for ViT/LIME
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
    input_tensors, labels, _ = next(iter(dl))
    
    # 2. Prepare Inputs for Model and Explainer
    if modality_type in ["XRAY", "ULTRASOUND"]:
        # 2D case: Model expects a list of PIL images
        pil_imgs = preprocess_2d_for_pil(input_tensors)
        model_input = pil_imgs
        explainer_input = np.array(pil_imgs[0]) # First sample for LIME
    else:
        # 3D case: Model expects a list of Tensors (D, H, W)
        model_input = input_tensors
        explainer_input = input_tensors[0].cpu().numpy() # First sample for LIME
        
    # 3. Prediction and Evaluation
    probs = multimodal_model.predict(model_input, modality_type)
    preds = probs.argmax(axis=1)
    
    # Simple Mock Evaluation
    report = evaluate_model(labels.cpu().numpy()[:sample_count], preds, labels=[0, 1])
    show_metrics(report, title=f"Classification Report for {modality_type}")
    
    # 4. Explainability (Routing)
    print(f"\n--- Explanation for first sample (pred: {preds[0]}) ---")

    if modality_type in ["XRAY", "ULTRASOUND"]:
        # 2D Explainer (LIME) setup
        def predict_wrapper_2d(images_np):
            # images_np: list or array HxWxC (RGB uint8) -> PIL list for 2D model
            pil_list = [Image.fromarray(x.astype('uint8')) for x in images_np]
            return multimodal_model.predict(pil_list, modality_type)
            
        lime = LIME2DExplainer(predict_wrapper_2d)
        explanation, temp, mask = lime.explain(explainer_input, label=preds[0], num_features=10)
        show_image_with_mask(explainer_input, mask, title=f"LIME 2D Mask for {modality_type} (pred {preds[0]})")
        
    elif modality_type in ["CT", "MRI"]:
        # 3D Explainer (Conceptual LIME) setup
        def predict_wrapper_3d(volumes_np):
            # volumes_np: list or array (D, H, W) -> torch tensor list for 3D model
            volume_tensors = [torch.from_numpy(v).float() for v in volumes_np]
            return multimodal_model.predict(volume_tensors, modality_type)
        
        lime = LIME3DExplainer(predict_wrapper_3d)
        _, volume_out, mask_out = lime.explain(explainer_input, label=preds[0])
        show_3d_explanation(volume_out, mask_out, title=f"LIME 3D Mask for {modality_type} (pred {preds[0]})")


def main_demo():
    print("Initializing Multimodal Model Router...")
    multimodal_vit = MultimodalViTWrapper()

    # Run demos for various modalities
    demo_modality(modality_type="XRAY", dataset_name="chestmnist", multimodal_model=multimodal_vit, sample_count=4)
    demo_modality(modality_type="CT", dataset_name="none", multimodal_model=multimodal_vit, sample_count=2)

if __name__ == "__main__":
    main_demo()
