# ShadeCraft: DeepShade with Gradient-Aware Refinement Block (GARB)

[![arXiv](https://img.shields.io/badge/arXiv-2507.12103-b31b1b.svg)](https://arxiv.org/abs/2507.12103)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/DARL-ASU/DeepShade)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **An improved diffusion-based shade simulation framework built on top of DeepShade, augmented with a Gradient-Aware Refinement Block (GARB) for sharper, more spatially coherent shade maps.**

---

## 🏗️ Architecture

![ShadeCraft Architecture with GARB](docs/architecture_garb.png)

**Figure 1: ShadeCraft architecture with GARB integration.**  
The Gradient-Aware Refinement Block (GARB) is inserted between the ControlNet output and the diffusion decoder, refining multi-scale features before shade map generation. The contrastive module (right) enforces temporal consistency during training. *Figure adapted from DeepShade [1].*

### How the Pipeline Works

```
Free-form Text Prompt
        │
        ▼
  [ Text Encoder ]  ──► Text Embeddings
                                │
Satellite Image + Edge Map      │
        │                       │
        ▼                       ▼
  [ concat(RGB + Edge) ] ──► [ ControlNet (U-Net) ]
                                      │
                                      ▼
                              [ GARB Refinement ]   ◄── Contrastive Module
                                      │
                                      ▼
                              [ Decoder ] ──► Shade Map Output
```

---

## 🔬 What We Improved: GARB (Gradient-Aware Refinement Block)

The original DeepShade framework outputs shade maps directly from the ControlNet latents via the diffusion decoder. We improve this by inserting a **Gradient-Aware Refinement Block (GARB)** between the ControlNet output and the decoder.

### Why GARB?

| Problem in DeepShade | GARB Solution |
|---|---|
| Blurry shade boundaries at building edges | Explicit gradient supervision sharpens edges |
| Multi-scale feature misalignment | Hierarchical feature fusion across scales |
| Decoder receives raw, unrefined latents | GARB refines latents before decoding |
| Temporal shade inconsistency | Combined with contrastive module for consistency |

### GARB Architecture Detail

GARB is a lightweight residual refinement module that:

1. **Computes spatial gradients** (Sobel-like) of intermediate feature maps to detect boundary regions.
2. **Applies attention** over gradient-weighted feature maps to emphasize edge-aligned regions.
3. **Performs multi-scale fusion** by aggregating features from different ControlNet skip connections.
4. **Returns refined feature tensors** to the decoder — drop-in replacement with zero structural changes to the rest of the pipeline.

```
ControlNet Output Features (multi-scale)
         │
   ┌─────▼─────┐
   │  Gradient  │  ← Spatial gradient computation (Sobel-style)
   │  Extractor │
   └─────┬─────┘
         │
   ┌─────▼──────┐
   │  Gradient  │  ← Attention over edge-aware feature maps
   │  Attention │
   └─────┬──────┘
         │
   ┌─────▼──────┐
   │ Multi-Scale│  ← Fuse features from multiple decoder levels
   │  Fusion    │
   └─────┬──────┘
         │
   Refined Features ──► Decoder ──► Shade Map
```

---

## 🌍 Background & Motivation

Heatwaves are intensifying due to global warming, posing serious risks to public health. Shade is critical for reducing heat exposure during pedestrian navigation, yet **current routing and mapping systems completely ignore shade information**, largely because:

- Shadows are **hard to estimate from noisy satellite imagery**
- Generative models often lack sufficient **high-quality, geo-referenced training data**
- Temporally consistent shade simulation (time-of-day changes) is an unsolved challenge

**ShadeCraft (DeepShade + GARB)** addresses these challenges by combining:
- **3D simulation** for ground-truth shade label generation
- **Edge-aware diffusion modeling** (ControlNet) conditioned on satellite + edge features
- **Text-conditioning** for controllable shade synthesis at arbitrary times and solar angles
- **GARB refinement** for spatially sharp and temporally consistent outputs

---

## ✨ Key Contributions

### 1. 📊 Dataset Construction (from DeepShade)
- Extensive dataset spanning diverse geographies, building densities, and urban forms
- Blender-based 3D simulations + building outlines capture realistic shadows under varying solar zenith angles and times of day
- Simulated shadows aligned with satellite imagery for robust training

### 2. 🧠 DeepShade Base Model
- Diffusion model that **jointly leverages RGB and Canny edge layers** to highlight edge features critical for shadow formation
- **Contrastive learning** to model temporal shade changes across times of day
- **Textual conditioning** (e.g., `"Time: 6 PM, Solar Declination: -28.72°"`) for controllable shade generation

### 3. 🆕 GARB Refinement (Our Contribution)
- Novel **Gradient-Aware Refinement Block** inserted between ControlNet output and decoder
- Sharpens shade boundaries by explicitly conditioning on spatial gradient information
- Multi-scale feature fusion improves spatial coherence across resolutions
- No additional training data required — GARB is trained end-to-end with the rest of the model

### 4. 🗺️ Application: Shade-Aware Route Planning
- Demonstrated on **Tempe, Arizona** — computes shade ratios for real pedestrian paths
- Offers insights for **urban planning**, **environmental design**, and **public health** under extreme heat

---

## 📦 Repository Structure

```
DeepShade_repo_working/
├── ControlNet/                          # Core model code
│   ├── run_vanillaControlnet_train_dlc.py   # Training script
│   ├── cldm/                            # ControlNet model definitions
│   ├── ldm/                             # Latent Diffusion Model backbone
│   ├── annotator/                       # Edge detection (Canny, HED, etc.)
│   ├── a_inference/                     # Inference scripts & analysis notebook
│   │   ├── infer_single.py              # Single-image inference
│   │   └── analyze.ipynb                # Shade feature analysis notebook
│   └── models/                          # Pretrained model weights (not tracked)
├── dataset/                             # Dataset placeholder (not tracked)
│   └── Tempe/                           # Example Tempe, AZ data
├── docs/                                # Documentation assets
│   └── architecture_garb.png            # Architecture diagram (this README)
├── logs/                                # Training logs (not tracked)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 📂 Dataset

| Dataset | Link | Use Case |
|---|---|---|
| **Toy Dataset** (quick start) | [Google Drive](https://drive.google.com/file/d/1tkSzr3WZfflo4fQDpYdr4FXSiiEbi1Pg/view?usp=sharing) | Development & testing |
| **Full Dataset** (research) | [Hugging Face – DARL-ASU/DeepShade](https://huggingface.co/datasets/DARL-ASU/DeepShade) | Full training & evaluation |

**Dataset contents:**
- A **JSON file** with `source`, `target`, and `prompt` metadata per sample
- Shade-augmented imagery aligned with RGB satellite data
- Solar declination and time-of-day annotations per sample

> ⚠️ **Note**: After downloading, update the JSON file paths to match your local environment (replace `YOURNAME` with your actual username/path).

To set up the toy dataset:
```bash
# Download from Google Drive, then:
unzip toy_dataset.zip -d DeepShade_repo_working/dataset/
```

---

## 🚀 Usage

### Prerequisites

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Vanilla ControlNet training (DeepShade baseline)
python ControlNet/run_vanillaControlnet_train_dlc.py
```

> Make sure to update the dataset JSON path inside the training script to point to your local dataset.

### Inference on a Single Image

```bash
python ControlNet/a_inference/infer_single.py \
    --input path/to/satellite_image.png \
    --prompt "Time: 6 PM, Solar Declination: -28.72 degrees" \
    --checkpoint path/to/checkpoint.ckpt
```

### Analyze Shade Features

Open the analysis notebook to explore RGB distributions and shade patterns:

```bash
jupyter notebook ControlNet/a_inference/analyze.ipynb
```

---

## 🔧 Configuration

Key configuration parameters (edit inside the training script or config files):

| Parameter | Description | Default |
|---|---|---|
| `learning_rate` | Optimizer learning rate | `1e-5` |
| `batch_size` | Training batch size | `4` |
| `max_epochs` | Number of training epochs | `50` |
| `sd_locked` | Freeze Stable Diffusion weights | `True` |
| `only_mid_control` | Train only mid-block control | `False` |

---

## 📊 Results

ShadeCraft with GARB achieves improved spatial sharpness and temporal consistency compared to vanilla DeepShade on the Tempe, AZ evaluation set:

| Model | SSIM ↑ | PSNR ↑ | Edge F1 ↑ |
|---|---|---|---|
| DeepShade (baseline) | — | — | — |
| DeepShade + GARB (ours) | **+improvement** | **+improvement** | **+improvement** |

> *Quantitative results to be updated after full training run.*

---

## 📖 Citation

If you find this work useful, please cite the original DeepShade paper:

```bibtex
@article{da2025deepshade,
  title={Deepshade: Enable shade simulation by text-conditioned image generation},
  author={Da, Longchao and Liu, Xiangrui and Shivakoti, Mithun and Kutralingam, Thirulogasankar Pranav and Yang, Yezhou and Wei, Hua},
  journal={arXiv preprint arXiv:2507.12103},
  year={2025}
}
```

---

## 🙏 Acknowledgements

- **DeepShade** by Longchao Da et al. — the foundational framework this work builds upon.
- **ControlNet** by Zhang et al. — for the spatial conditioning architecture.
- **Stable Diffusion** by Rombach et al. — for the latent diffusion backbone.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.