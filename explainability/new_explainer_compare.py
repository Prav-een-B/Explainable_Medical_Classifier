"""
explain_compare.py
------------------
Standalone explainability script for MultiTaskViT:
Generates GradCAM, LIME, and SHAP visualizations side-by-side.

Requirements:
pip install torch torchvision timm lime shap scikit-image pytorch-grad-cam matplotlib pillow numpy
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.multitask_vit import MultiTaskViT
import timm
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------- CONFIG ----------------------
CKPT_PATH = r"C:\Users\Yuvi\Downloads\vit_epoch12_skin_mri_only.pt"      # your .pt file path
IMG_PATH = "glioma_tumor_0870.jpg"      # input image path
MODALITY = "MRI"                 # one of: "XRAY", "SKIN", "MRI"
OUT_DIR = Path("outputs")         # folder to save visualizations
OUT_DIR.mkdir(exist_ok=True)
# ----------------------------------------------------


#model
        
def vit_reshape_transform(tensor, height=14, width=14):
    # tensor shape: [B, Tokens, D]
    # skip class token
    tensor = tensor[:, 1:, :]
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result.contiguous()



# ---------------------- LOADER ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_xray, n_skin, n_mri = 14, 7, 2
model = MultiTaskViT(n_xray, n_skin, n_mri)
ckpt = torch.load(CKPT_PATH, map_location=device)
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()
print("✅ Model loaded.")

# ---------------------- IMAGE ----------------------
pil_img = Image.open(IMG_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
input_tensor = transform(pil_img).unsqueeze(0).to(device)
rgb_img = np.array(pil_img.resize((224,224))) / 255.0

# ---------------------- 1️⃣ GRADCAM ----------------------
# ----- WE MUST USE A GRADIENT-FREE METHOD LIKE EIGENCAM -----

from pytorch_grad_cam import EigenCAM  # <--- FIX 1: Use EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

vit_model = model.vit
if hasattr(vit_model, "encoder"):
    target_layers = [vit_model.encoder.layer[-1].output]
    print("ℹ️ Using HuggingFace ViT encoder output for GradCAM (pytorch-grad-cam)")
else:
    target_layers = [vit_model.blocks[-1].mlp] # Using .mlp is still correct
    print("ℹ️ Using timm ViT last block 'mlp' for GradCAM")

# Wrap model for GradCAM so it always uses the right task
class ModelForGradCAM(nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task
    def forward(self, x):
        return self.model(x, self.task)

wrapped_model = ModelForGradCAM(model, MODALITY)

# --- FIX 2: Instantiate EigenCAM instead of GradCAMPlusPlus ---
cam = EigenCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=vit_reshape_transform)

outputs = model(input_tensor, MODALITY)
if isinstance(outputs, (tuple, list)):
    outputs = outputs[0]

pred = torch.argmax(outputs, dim=1).item()
print(f"ℹ️ Model predicted class: {pred} for modality {MODALITY}")

# We still use ClassifierOutputTarget, as it helps EigenCAM focus
targets = [ClassifierOutputTarget(pred)]

# This call will now work without gradients
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

gradcam_vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
gradcam_img = Image.fromarray(gradcam_vis)
gradcam_path = OUT_DIR / "gradcam.png"
gradcam_img.save(gradcam_path)
print(f"✅ GradCAM saved → {gradcam_path}")


# ---------------------- 2️⃣ LIME ----------------------
class ExplainWrapper:
    def __init__(self, model, task, device):
        self.model = model
        self.task = task
        self.device = device
        self.model.eval()

    def __call__(self, imgs):
        with torch.no_grad():
            tensor = torch.tensor(imgs).permute(0, 3, 1, 2).float() / 255.0
            tensor = torch.nn.functional.interpolate(tensor, size=(224, 224))
            tensor = torch.stack([
                (x - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) /
                torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                for x in tensor
            ])
            tensor = tensor.to(self.device)
            logits = self.model(tensor, self.task)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
from lime.wrappers.scikit_image import SegmentationAlgorithm

segmenter = SegmentationAlgorithm(
        'slic',
        n_segments=100,        # try 50–150 for MRI
        compactness=10,        # controls how square vs irregular segments are
        sigma=1
    )

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    np.array(pil_img.resize((224,224))),
    classifier_fn=ExplainWrapper(model, MODALITY, device),
    top_labels=1,
    hide_color=None,
    num_samples=4000,
    segmentation_fn=segmenter
)
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)
temp = (temp-temp.min()) / (temp.max()-temp.min())
lime_img = mark_boundaries(temp, mask)
lime_path = OUT_DIR / "lime.png"
plt.imsave(lime_path, lime_img)
print(f"✅ LIME saved → {lime_path}")


# ---------------------- 3️⃣ SHAP ----------------------
import shap

np_img = np.array(pil_img.resize((224,224))) / 255.0
background = np.expand_dims(np_img, 0)

def predict_fn(imgs):
    imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float().to(device)
    imgs = torch.nn.functional.interpolate(imgs, size=(224, 224))
    imgs = (imgs - torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]) / \
           torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with torch.no_grad():
        preds = model(imgs, MODALITY)
        return torch.softmax(preds, dim=1).cpu().numpy()

masker = shap.maskers.Image("inpaint_telea", (224, 224, 3))

print("⚙️ Running SHAP (this may take 1–3 minutes)...")
explainer = shap.Explainer(predict_fn, masker, output_names=[f"Class {i}" for i in range(14)])
shap_values = explainer(background)

shap.image_plot(shap_values, background, show=False)
plt.savefig(OUT_DIR / "shap.png", bbox_inches='tight')
plt.close()
shap_path = OUT_DIR / 'shap.png'
print(f"✅ SHAP saved → {OUT_DIR / 'shap.png'}")


# ---------------------- 4️⃣ COMPARISON GRID ----------------------
fig, axs = plt.subplots(1, 4, figsize=(16, 5))
axs[0].imshow(pil_img)
axs[0].set_title("Original")
axs[1].imshow(gradcam_img)
axs[1].set_title("GradCAM")
axs[2].imshow(lime_img)
axs[2].set_title("LIME")
axs[3].imshow(plt.imread(shap_path))
axs[3].set_title("SHAP")

for ax in axs:
    ax.axis("off")

final_path = OUT_DIR / "comparison_grid.png"
plt.tight_layout()
plt.savefig(final_path, bbox_inches="tight")
plt.close()
print(f"✅ Comparison grid saved → {final_path}")
