# ZTOF
Implementation code for the ZTOF framework.
# ZTOF: A Closed-Loop Optimization Framework for Fine-Grained Visual Recognition

Welcome to the official GitHub repository for **ZTOF** (Zero-Shot Text Optimization Framework), a novel approach for enhancing fine-grained visual recognition through closed-loop image-text alignment optimization. This repository contains the source code, datasets, and supplementary materials for our research.


## 📌 Overview
ZTOF addresses the challenge of distinguishing visually similar categories (e.g., food items, bird species, or insect morphologies) by iteratively refining text descriptions and aligning them with visual features. By leveraging CLIP for vision-language alignment and large language models (LLMs) for text refinement, our framework achieves state-of-the-art performance on fine-grained recognition tasks.

### Key Features:
- **Closed-Loop Optimization**: Iteratively improves text descriptions to better match visual features.
- **Fine-Grained Discrimination**: Effectively distinguishes between visually similar categories (e.g., pho vs. hot and sour soup).
- **Generalization**: Works across diverse datasets (Food-101, CUB-200-2011, Pet Dataset, and private mosquito datasets).


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
   git clone https://github.com/xxx/recognition-framework.git
   cd recognition-framework
