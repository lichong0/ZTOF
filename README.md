# ZTOF
Implementation code for the ZTOF framework.
# From Zero to One: Accelerating Recognition Model Construction via Self-Optimizing Prototyping Mechanism

Welcome to the official GitHub repository for **ZTOF** (Zero-Shot Text Optimization Framework), a novel approach for enhancing fine-grained visual recognition through closed-loop image-text alignment optimization. This repository contains the source code, datasets, and supplementary materials for our research.


## 📌 Overview
ZTOF addresses the challenge of distinguishing visually similar categories (e.g., food items, bird species, or insect morphologies) by iteratively refining text descriptions and aligning them with visual features. By leveraging CLIP for vision-language alignment and large language models (LLMs) for text refinement, our framework achieves state-of-the-art performance on fine-grained recognition tasks.

### Key Features:
- **Closed-Loop Optimization**: Iteratively improves text descriptions to better match visual features.
- **Fine-Grained Discrimination**: Effectively distinguishes between visually similar categories (e.g., pho vs. hot and sour soup).
- **Generalization**: Works across diverse datasets (Food-101, CUB-200-2011, Pet Dataset, and private mosquito datasets).
  
## 📊 Dataset Description

### Public Datasets
- **Food-101** https://www.vision.caltech.edu/datasets/cub_200_2011/
- **CUB-200-2011** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Pet Dataset** https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
### Private Dataset
A partial sample (20 images) of our private mosquito dataset is available in `datasets/private_mosquito_samples/`, with sensitive metadata (collection locations and timestamps) removed. The complete dataset contains 700 images across 30 mosquito species, focusing on key morphological features (wing veins, body shape, etc.).

## 📁 Repository Structure




## 🚀 Getting Started

### Prerequisites
- Python 3.12.3
- PyTorch 2.3.0
- NVIDIA GPU (RTX 4090 recommended for efficient training)
- CUDA 11.8+ (for GPU acceleration)


### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/lichong0/ZTOF.git
   cd ZTOF
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. (Optional) Download pre-trained models:
   - CLIP ViT-B/32 (automatically downloaded by the code)
   - Qwen2.5-7B
4. train
   ```bash
   python train.sh
  
