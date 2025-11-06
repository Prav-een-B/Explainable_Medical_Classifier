# utils/visualization.py
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
import torch

def show_image(image_np, title="Image"):
    # ... (function unchanged) ...

def show_image_with_mask(image_np, mask, title="Explanation Mask"):
    # ... (function unchanged) ...

def show_3d_explanation(volume_tensor, heatmap_tensor, title="3D Grad-CAM Explanation"):
    """
    Visualizes a 3D Grad-CAM heatmap overlaid on a central slice 
    of the original 3D volume.
    """
    # Ensure tensors are on CPU and numpy
    volume_np = volume_tensor.cpu().numpy().squeeze() # (D, H, W)
    heatmap_np = heatmap_tensor.cpu().numpy().squeeze() # (D, H, W)

    # Get central slice index
    center_slice_idx = volume_np.shape[0] // 2
    
    volume_slice = volume_np[center_slice_idx, :, :]
    heatmap_slice = heatmap_np[center_slice_idx, :, :]

    plt.figure(figsize=(12, 6))
    
    # Show original slice
    plt.subplot(1, 2, 1)
    plt.imshow(volume_slice, cmap='gray')
    plt.title(f"Original Volume (Slice {center_slice_idx})")
    plt.axis('off')
    
    # Show heatmap overlaid on slice
    plt.subplot(1, 2, 2)
    plt.imshow(volume_slice, cmap='gray')
    plt.imshow(heatmap_slice, cmap='jet', alpha=0.5) # Overlay heatmap
    plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_metrics(report, title="Model Evaluation"):
    # ... (function unchanged) ...
