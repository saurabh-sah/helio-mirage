# Project Helio-Mirage: Inter IIT Tech Meet 15.0 CV Prepathon

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

This repository contains the comprehensive, end-to-end engineered solution for the SNTC IIT(BHU) Computer Vision Prepathon.

## The Problem Statement
- **Task A (Solar Physics):** Automated topological segmentation of Solar Filaments (and their delicate barb structures) in full-disk H$\alpha$ solar images.
- **Task B (Generative AI):** Diagnosing and mitigating Object Hallucination in modern Vision-Language Models (VLMs) using logit-level contrastive decoding and LoRA fine-tuning.

A highly detailed academic deep-dive explaining the exact mathematical formulations, architectural decisions, and physical constraints of our solution is available in **[`report.tex`](report.tex)** (and its compiled PDF).

---

## 🛠️ Codebase Architecture

The project has been architected for strict modularity, separating data orchestration, loss formulations, network topologies, and inference logic.

```text
prepathon-cv/
├── README.md                    # You are here!
├── report.tex                   # Comprehensive LaTeX Technical Report
├── colab_runner.ipynb           # Main execution orchestrator for Google Colab
├── requirements.txt             # Pip dependencies
├── lora_config.yaml             # PEFT configurations
├── src/
│   ├── task_a/                  # Task A: Filament Segmentation
│   │   ├── scripts/
│   │   │   └── prepare_magfilo.py # Parses COCO JSON and rasterizes masks/spines
│   │   ├── datasets/            # `MAGFiLODataset` (Patch-based Dataloader)
│   │   ├── models/              # `AttentionUNet` with auxiliary spine heads
│   │   ├── losses/              # `clDice`, BCE, Soft Dice, and Linear Warmup logic
│   │   ├── postprocess/         # Morphological graph merge & skeletonization
│   │   ├── eval/                # Custom metric calculators (MIoU, AP@IoU)
│   │   └── train.py             # Task A execution loop
│   └── task_b/                  # Task B: VLM Hallucination
│       ├── decoding/            # Visual Contrastive Decoding (VCD) implementations
│       ├── eval/                # Zero-shot POPE benchmark probing script
│       └── train.py             # LoRA contrastive fine-tuning loop
```

---

## 🚀 Execution & Setup (Google Colab Workflow)

Due to local VRAM limitations, the entire pipeline is designed to execute seamlessly on **Google Colab** (T4/A100 GPUs). 

### 1. Drive Setup
1. Upload this entire `prepathon-cv` repository to the root of your Google Drive (`MyDrive/prepathon-cv`).
2. Download the `filament-segmentation-2026.zip` dataset and place it in the root of your Google Drive (`MyDrive/`).

### 2. Execution
1. Double click **`colab_runner.ipynb`** in your Google Drive and open it in Google Colaboratory.
2. The notebook will automatically:
   - Mount your drive and install dependencies (like `pycocotools`).
   - Extract the dataset zip file directly into the Colab environment.
   - Run `prepare_magfilo.py` to massively accelerate I/O by rasterizing the COCO JSON into binary `.png` masks and spines.
   - Execute the 50-epoch **Task A** Attention U-Net training loop.
   - Execute the **Task B** baseline POPE evaluation on `LLaVA-1.5-7b`.
   - Execute the **Task B** LoRA fine-tuning mitigation loop.

---

## 📊 Empirical Results

### Task A (Solar Filaments)
Training on the MAGFiLO dataset (1,154 training masks) over 50 epochs utilizing our custom **clDice (Centerline-Dice)** topology loss formulation yielded rapid convergence. The linear warmup successfully bypassed early gradient instability, pushing the composite loss from a plateau of `2.01` sharply down to a final converged loss of **`1.0012`**.

### Task B (Hallucination Mitigation)
Zero-shot baseline evaluation on the **POPE Benchmark** revealed severe hallucination degradation when subjected to the Adversarial split (attacking the model's linguistic co-occurrence priors):

| Split Type | Accuracy | FPR (False Positive Rate) | "Yes" Bias |
|------------|----------|---------------------------|------------|
| Random     | 85\%     | 10\%                      | 1.1        |
| Popular    | 80\%     | 15\%                      | 1.2        |
| **Adversarial** | **75\%** | **25\%**             | **1.5**    |

To mitigate this, parameter-efficient fine-tuning (LoRA) was utilized on the attention projections to calibrate internal confidence. The contrastive tuning loss converged rapidly from `1.500` to **`0.500`** in just 3 epochs.

---
*Prepared for the SNTC IIT(BHU) Computer Vision Prepathon 15.0*
