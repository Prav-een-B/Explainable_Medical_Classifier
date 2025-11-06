# explainability/gradcam_explainer.py
"""
3D Explainability using Grad-CAM for MONAI models.
"""
import torch
from monai.visualize import GradCAM

# FIXED (Fix 5): Helper function to resolve string path to a module
def resolve_module(root, path):
    """
    Resolves a string path (e.g., 'transformer.blocks[-1].norm1') 
    to an actual module object within the root model.
    """
    if path.startswith("model."):
        path = path[len("model."):]
    obj = root
    for part in path.split('.'):
        if '[' in part and ']' in part:
            name, idxpart = part.split('[')
            idx = int(idxpart.strip(']'))
            obj = getattr(obj, name)[idx]
        else:
            obj = getattr(obj, part)
    return obj

class GradCAM3DExplainer:
    def __init__(self, model, target_layer):
        """
        model: The MONAI 3D model (e.g., ViT)
        target_layer: The layer name string to hook into. 
                      (e.g., 'transformer.blocks[-1].norm1')
        """
        self.model = model
        
        # FIXED (Fix 5): Resolve the string path to the actual module
        if isinstance(target_layer, str):
            target_mod = resolve_module(self.model, target_layer)
        else:
            target_mod = target_layer
            
        self.gradcam = GradCAM(
            nn_module=self.model,
            target_layers=[target_mod] # MONAI expects a list of modules
        )

    def explain(self, volume_tensor_batch, label=None):
        """
        Generates a 3D heatmap for the batch.
        volume_tensor_batch: A batch of 3D volumes (B, C, D, H, W)
        label: The class index to explain for. If None, explains the predicted class.
        
        Returns: A 3D heatmap (B, 1, D, H, W)
        """
        # FIXED (Fix 6): Use model.eval() with gradients enabled
        self.model.eval()
        with torch.enable_grad():
            # (Fix 4, Potential Issue): Ensure tensor is on the same device as the model
            volume_tensor_batch = volume_tensor_batch.to(next(self.model.parameters()).device)
            
            heatmap = self.gradcam(x=volume_tensor_batch, class_idx=label)
            
        return heatmap
