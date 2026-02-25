# Hybridiff: Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling

*Official implementation of the CVPR 2026 paper "Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling”*

**Hybridiff** is a high-performance diffusion model inference framework that combines condition-based partitioning with adaptive parallelism switching for accelerated image generation. This repository implements dynamic threshold detection (`tau1`, `tau2`) to automatically optimize the trade-off between quality and speed during the diffusion inference.

## Installation

### 1. Create Virtual Environment

```bash
# Create and activate conda environment
conda create -n hybridiff python=3.10
conda activate hybridiff
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

## Dataset Preparation

### MS-COCO Dataset

For large-scale evaluation using MS-COCO dataset:

```bash
# The dataset will be automatically downloaded when running generation scripts
# Default cache location: /data/ms-coco
# You can modify this path in the generation scripts if needed
```

**Note**: The hardcoded dataset path in the generation scripts (`generate_hybridiff_coco.py` and `generate_sd3_hybridiff_coco.py`) may need to be updated based on your environment.

## Quick Start: Single Image Generation

### Stable Diffusion XL (SDXL)

```bash
# Run with 2 GPUs
torchrun --nproc_per_node=2 examples/run_sdxl_hybridiff.py \
  --model_n 2 --stride 1 \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --L_slope 12 --eps_slope 0.0004 --k 5 --tau_cap 15
```

### Stable Diffusion 3 (SD3)

```bash
# Run with 2 GPUs
torchrun --nproc_per_node=2 examples/run_sd3_hybridiff.py \
  --model_n 2 --stride 1 \
  --prompt "A cat holding a sign that says hello world" \
  --L_slope 15 --eps_slope 0.0005 --k 5 --tau_cap 40
```

## COCO Dataset Generation

### SDXL on COCO

```bash
# Generate images for COCO validation set
torchrun --nproc_per_node=2 examples/generate_hybridiff_coco.py \
  --model_n 2 --stride 1 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --scheduler ddim \
  --L_slope 12 --eps_slope 0.0004 --k 5 --tau_cap 15

# Or use the provided shell script
bash examples/run_hybridiff_coco.sh
```

### SD3 on COCO

```bash
# Generate images for COCO validation set
torchrun --nproc_per_node=2 examples/generate_sd3_hybridiff_coco.py \
  --model_n 2 --stride 1 \
  --num_inference_steps 50 \
  --guidance_scale 7.0 \
  --L_slope 15 --eps_slope 0.0005 --k 5 --tau_cap 40

# Or use the provided shell script
bash examples/run_sd3_hybridiff_coco.sh
```

## Metric Evaluation

Compute LPIPS, PSNR, and FID metrics between generated and reference images:

```bash
python examples/compute_metric.py \
  --input_root0 path/to/reference/images \
  --input_root1 path/to/generated/images \
  --is_gt \
  --max_length 5000 \
```

## Key Parameters

### Model Configuration (Based on asyncdiff)

- **`model_n`**: Number of GPUs to use for parallelization (2 or 4)
  - For `world_size=2`: set `model_n=2`, `stride=1`
- **`stride`**: Stride for pipeline parallelism (1 or 2) -> Based on asyncdiff
- **`time_shift`**: Enable time-shifted inference (default: False) -> Based on asyncdiff

### Hybridiff Threshold Parameters

- **`L_slope`**: Window size for computing slope in tau1 detection
  - SDXL default: 12
  - SD3 default: 15
  - Larger values result in smoother detection but slower response

- **`eps_slope`**: Threshold for slope-based tau1 detection
  - SDXL default: 0.0004 (0.4e-3)
  - SD3 default: 0.0005 (0.5e-3)
  - Lower values trigger tau1 earlier (more aggressive parallelism)

- **`k`**: Offset for tau2 calculation (`tau2 = tau1 + k`)
  - SDXL default: 5
  - SD3 default: 5
  - Determines the length of parallel execution phase

- **`tau_cap`**: Safety fallback value for tau1 if automatic detection fails
  - SDXL default: 15
  - SD3 default: 40
  - Ensures tau1 is set even if slope-based detection doesn't trigger

### Generation Parameters

- **`num_inference_steps`**: Number of denoising steps (default: 50)
- **`guidance_scale`**: CFG guidance scale
  - SDXL default: 5.0
  - SD3 default: 7.0
- **`image_size`**: Output image resolution (default: 1024)
- **`scheduler`**: Noise scheduler type (for SDXL: "euler", "dpm-solver", or "ddim" -> default: "ddim")

## Structure

```
hybridiff/
├── hybrid_sd.py              # HybriDiff implementation for SDXL
├── hybrid_sd3.py             # HybriDiff implementation for SD3
├── pipelines/
│   ├── __init__.py
│   ├── pipeline_stable_diffusion_xl.py     # Custom SDXL pipeline with CFG-DP
│   └── pipeline_stable_diffusion_3.py      # Custom SD3 pipeline with CFG-DP
├── metrics.py                # Memory and performance measurement utilities
├── tools.py                  # Helper utilities for result caching
└── pipe_config.py            # Model splitting configuration

examples/
├── run_sdxl_hybridiff.py     # Single image generation (SDXL)
├── run_sd3_hybridiff.py      # Single image generation (SD3)
├── generate_hybridiff_coco.py       # COCO dataset generation (SDXL)
├── generate_sd3_hybridiff_coco.py   # COCO dataset generation (SD3)
├── compute_metric.py         # Metric evaluation script
└── *.sh                      # Shell scripts for batch execution
```

## License

This project is licensed under the Apache License 2.0 - see the original Diffusers license for details.

## Acknowledgments

- Built on top of [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Inspired by [AsyncDiff](https://github.com/czg1225/AsyncDiff) and other parallel diffusion inference works
- Thanks to the Stable Diffusion community for pre-trained models

## Citation

If you use this research, please cite our paper:

```bibtex
@inproceedings{jung2026hybridiff,
  title={Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling},
  author={Euisoo Jung, Byunghyun Kim, Hyunjin Kim, Seonghye Cho and Jae-Gil Lee},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
