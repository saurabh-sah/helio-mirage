# Inter IIT Tech Meet 15.0 — Computer Vision Prepathon

This repository contains the solution for the SNTC IIT(BHU) Computer Vision Prepathon.
It includes implementations for:
- **Task A (Compulsory):** Automated Segmentation of Solar Filaments in full-disk Hα images.
- **Task B:** Diagnosing Object Hallucination in Vision-Language Models.

## Codebase and File Structure

```text
prepathon-cv/
├── README.md                    # This file
├── report/                      # Final report and figures
├── configs/                     # Hydra configuration files
│   ├── task_a/
│   └── task_b/
├── data/                        # Datasets (raw and processed) - ignored in git
├── src/
│   ├── task_a/                  # Task A implementation (solar filaments)
│   │   ├── preprocessing/       # Disk detection, limb correction, etc.
│   │   ├── datasets/            # Dataset loaders
│   │   ├── models/              # U-Net with attention gates
│   │   ├── losses/              # clDice, BCE, Dice
│   │   ├── postprocess/         # Skeletonization, graph merge
│   │   ├── eval/                # IoU, AP, Hit/Miss, MIoU
│   │   ├── train.py             # Training entry point
│   │   └── infer.py             # Inference entry point
│   ├── task_b/                  # Task B implementation (hallucinations)
│   │   ├── decoding/            # Decoding strategies (VCD, OPERA)
│   │   ├── lora/                # LoRA fine-tuning
│   │   ├── diagnostics/         # Hallucination diagnostics
│   │   ├── eval/                # POPE benchmark eval
│   │   ├── train.py
│   │   └── infer.py
│   └── common/                  # Shared utilities
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # Unit tests
└── environment.yml              # Conda environment file
```

## Setup Instructions
1. Install the conda environment: `conda env create -f environment.yml`
2. Activate environment: `conda activate prepathon_cv`
3. Download the MAGFiLO dataset to `data/raw/`.
4. Ensure `HF_TOKEN` and `WANDB_API_KEY` are set.
