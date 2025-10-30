# Explainable Medical Image Classifier

Prototype and codebase for building an explainable medical image classifier using Vision Transformers (ViT) and post-hoc explainability methods (LIME, SHAP).

## Structure
- `data/` : dataset loader and transforms
- `models/` : model wrappers (ViT)
- `explainability/` : LIME and SHAP explainer wrappers
- `utils/` : visualization and metrics
- `main.py` : demo script for inference + LIME explanation
- `config.py` : central configuration
- `requirements.txt` : necessary packages

## Quick start (Colab / local)
1. Clone:
```bash
git clone <repo-url>
cd explainable_medical_classifier
