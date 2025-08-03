# ZTOF
Implementation code for the ZTOF framework.
# From Zero to One: Accelerating Recognition Model Construction via Self-Optimizing Prototyping Mechanism

Welcome to the official GitHub repository for **ZTOF** (From Zero to One Framework), a framework designed for rapid prototyping of visual recognition models


## 📌 Overview
ZTOF proposes a self-optimizing mechanism that establishes a feedback loop between a large language model (LLM) and a pre-trained vision-language alignment model. By combining the knowledge reasoning capability of the LLM with perceptual feedback from the alignment model, ZTOF automatically generates and improves category-level textual descriptions without requiring manual annotations.

### Key Features:
- **Closed-Loop Optimization**: Iteratively improves text descriptions to better match visual features.
- **Fine-Grained Discrimination**: Effectively distinguishes between visually similar categories.
- **Generalization**: Works across diverse datasets (Food-101, CUB-200-2011, Pet Dataset, and private mosquito datasets).
  
## 📊 Dataset Description

### Public Datasets
- **Food-101** https://www.vision.caltech.edu/datasets/cub_200_2011/
- **CUB-200-2011** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Pet Dataset** https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
### Private Dataset
A partial sample (20 images) of our private mosquito dataset is available in `data/mosquito`, with sensitive metadata (collection locations and timestamps) removed. The complete dataset contains 700 images across 30 mosquito species, focusing on key morphological features (wing veins, body shape, etc.)

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
  
