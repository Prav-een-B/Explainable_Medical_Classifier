# main.py
"""
Simple orchestrator for loading data, running model inference and explainability.
This script demonstrates inference and LIME-based explanation on a few samples.
"""

import numpy as np
from PIL import Image
from data.dataset_loader import get_dataloader
from models.vit_model import ViTWrapper
from explainability.lime_explainer import LIMEExplainer
from utils.visualization import show_image, show_image_with_mask
from config import DEVICE

def batch_to_pil(batch_images):
    # batch_images: tensor N,C,H,W in [0,1] expected by transforms.ToTensor() normalization undone if needed.
    pil_imgs = []
    for img in batch_images:
        arr = img.permute(1,2,0).cpu().numpy()
        # convert from normalized [-1,1] or used in dataset - we assume ToTensor() used plain [0,1]
        arr = (arr * 255).astype('uint8')
        pil_imgs.append(Image.fromarray(arr))
    return pil_imgs

def main_demo():
    print("Loading model and dataloader...")
    vit = ViTWrapper()
    dl = get_dataloader("chestmnist", split="test", batch_size=8, shuffle=False)

    # Grab one batch
    for batch in dl:
        # medmnist returns (image, label) where image is HxW or CxHxW depending
        images, labels = batch[0], batch[1]
        # convert to PIL list (helper)
        pil_imgs = []
        for i in range(len(images)):
            img = images[i]
            # if single channel
            if img.shape[0] == 1:
                arr = img.squeeze(0).numpy()
                arr = (arr * 255).astype('uint8')
                pil_imgs.append(Image.fromarray(arr).convert("RGB"))
            else:
                arr = img.permute(1,2,0).numpy()
                arr = (arr * 255).astype('uint8')
                pil_imgs.append(Image.fromarray(arr))
        break

    # Prediction
    probs = vit.predict(pil_imgs)
    preds = probs.argmax(axis=1)
    print("Preds:", preds, "Probs:", probs)

    # Prepare a LIME explainer
    def predict_wrapper(images_np):
        # images_np: list or array HxWxC (RGB uint8)
        # ViTWrapper.predict expects PIL images
        from PIL import Image
        pil_list = [Image.fromarray(x.astype('uint8')) for x in images_np]
        return vit.predict(pil_list)

    lime = LIMEExplainer(predict_wrapper)
    # explain first sample
    sample_np = np.array(pil_imgs[0])
    explanation, temp, mask = lime.explain(sample_np, label=preds[0], num_features=10)
    show_image_with_mask(sample_np, mask, title=f"LIME Mask (pred {preds[0]})")

if __name__ == "__main__":
    main_demo()
