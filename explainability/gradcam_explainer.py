# explainability/gradcam_explainer.py
"""
3D Explainability using Grad-CAM for MONAI models.
"""
import torch
from monai.visualize import GradCAM

class GradCAM3DExplainer:
    def __init__(self, model, target_layer):
        """
        model: The MONAI 3D model (e.g., ViT)
        target_layer: The layer name to hook into. 
                      For MONAI ViT, 'model.transformer.blocks[-1].norm1' is a good choice.
        """
        self.model = model
        self.gradcam = GradCAM(
            nn_module=self.model,
            target_layers=target_layer
        )

    def explain(self, volume_tensor_batch, label=None):
        """
        Generates a 3D heatmap for the batch.
        volume_tensor_batch: A batch of 3D volumes (B, C, D, H, W)
        label: The class index to explain for. If None, explains the predicted class.
        
        Returns: A 3D heatmap (B, 1, D, H, W)
        """
        # GradCAM requires the model to be in training mode to hook properly
        self.model.train() 
        
        # MONAI's GradCAM computes the map
        # We get one map per item in the batch
        heatmap = self.gradcam(x=volume_tensor_batch, class_idx=label)
        
        self.model.eval() # Set model back to eval mode
        return heatmap
