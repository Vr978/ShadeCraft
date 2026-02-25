# ---------- Core (install PyTorch separately; see RUN_LOCAL.md) ----------
pytorch-lightning==1.9.5
omegaconf==2.3.0
einops==0.7.0
opencv-python==4.8.1.78
scipy==1.10.1
numpy==1.24.4
Pillow==10.0.1
tqdm==4.66.1
PyYAML==6.0.1
safetensors==0.4.2

# ---------- Vision / metrics ----------
scikit-image==0.21.0
scikit-learn==1.3.0
lpips==0.1.4
kornia==0.6.9

# ---------- Text / CLIP (likely used by the model config) ----------
transformers==4.38.2
open-clip-torch==2.20.0
timm==0.9.2

# ---------- Utilities ----------
addict==2.4.0
matplotlib==3.7.3
pandas==2.0.3
prettytable==3.9.0
albumentations==1.3.1
basicsr==1.4.2
wandb==0.16.6

# ---------- Optional / speedups (comment out if they fail to build) ----------
# xformers==0.0.16
# turbojpeg==0.6.1
# cityscapesscripts==2.2.2

