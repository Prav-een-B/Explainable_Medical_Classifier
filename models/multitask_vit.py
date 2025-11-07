# models/multitask_vit.py
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from config import VIT_PRETRAINED

class MultiTaskViT(nn.Module):
    def __init__(self, n_xray, n_skin, n_mri):
        super().__init__()
        # load pretrained ViT (HuggingFace)
        self.vit = ViTModel.from_pretrained(VIT_PRETRAINED)
        hidden = self.vit.config.hidden_size  # typically 768

        # heads
        self.head_xray = nn.Linear(hidden, n_xray)
        self.head_skin = nn.Linear(hidden, n_skin)
        self.head_mri  = nn.Linear(hidden, n_mri)

    def forward(self, x, task):
        # x: B x 3 x H x W
        out = self.vit(x).last_hidden_state[:, 0, :]  # CLS token
        if task == "XRAY":
            return self.head_xray(out)
        elif task == "SKIN":
            return self.head_skin(out)
        elif task == "MRI":
            return self.head_mri(out)
        else:
            raise ValueError("Unknown task")
