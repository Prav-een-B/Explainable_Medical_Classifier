import torch
import torch.nn as nn
from transformers import ViTModel
from config import VIT_PRETRAINED

class MultiTaskViT(nn.Module):
    def __init__(self, n_xray, n_skin, n_mri):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_PRETRAINED)
        hidden = self.vit.config.hidden_size
        self.head_xray = nn.Linear(hidden, n_xray)
        self.head_skin = nn.Linear(hidden, n_skin)
        self.head_mri  = nn.Linear(hidden, n_mri)

    def forward(self, x, task):
        feats = self.vit(x).last_hidden_state[:, 0]
        if task == "XRAY":
            return self.head_xray(feats)
        elif task == "SKIN":
            return self.head_skin(feats)
        elif task == "MRI":
            return self.head_mri(feats)
        else:
            raise ValueError(f"Unknown task: {task}")
