# Distributionally Robust Classification for Multi-source Unsupervised Domain Adaptation

This code implements the algorithm from the following paper:

> Seonghwi Kim, Sung Ho Jo, Wooseok Ha, and Minwoo Chae
> [Distributionally Robust Classification for Multi-source Unsupervised Domain Adaptation](https://arxiv.org/abs/2601.21315) 

## Abstract

Unsupervised domain adaptation (UDA) is a statistical learning problem when the distribution of training (source) data is different from that of test (target) data. In this setting, one has access to labeled data only from the source domain and unlabeled data from the target domain. The central objective is to leverage the source data and the unlabeled target data to build models that generalize to the target domain. Despite its potential, existing UDA approaches often struggle in practice, particularly in scenarios where the target domain offers only limited unlabeled data or spurious correlations dominate the source domain. To address these challenges, we propose a novel distributionally robust learning framework that models uncertainty in both the covariate distribution and the conditional label distribution. Our approach is motivated by the multi-source domain adaptation setting but is also directly applicable to the single-source scenario, making it versatile in practice. We develop an efficient learning algorithm that can be seamlessly integrated with existing UDA methods. Extensive experiments under various distribution shift scenarios show that our method consistently outperforms strong baselines, especially when target data are extremely scarce.


## Prerequisites
- Python >= 3.10.9
- numpy >= 1.26.4
- pandas >= 2.3.3
- torch >= 2.7.1 (CUDA 12.6)
- torchvision >= 0.22.1 (CUDA 12.6)
- tqdm >= 4.32.2

## Code Structure and File Descriptions
This repository is organized into three main components, each responsible for a distinct part of the experimental pipeline. The current implementation provides an example setup for SVHN → MNIST unsupervised domain adaptation, but the structure is designed to be easily extensible to other dataset pairs.

### Ours(ERM)_main.ipynb
This notebook serves as the entry point of the experiment.

Main responsibilities:
- Parse and set experimental hyperparameters.
- Load source and target datasets via get_loader.
- Define and train the neural network using ERM-based optimization.
- Evaluate the trained model on target validation and test sets.
- Log training statistics and results using utilities from utils.py.

Remarks:
- The notebook currently demonstrates the SVHN → MNIST setting.
- By changing the dataset argument (e.g., dset='u2m', dset='m2u') and corresponding paths, the same training logic can be reused for other domain adaptation benchmarks.
- The ERM baseline is trained from scratch using the same codebase (pretrained ERM checkpoints are not provided.)


### get_data_ours.py
Dataset loading and preprocessing

Main responsibilities:
- Download datasets automatically if not present.
- Apply dataset-specific preprocessing and transformations.
- Construct PyTorch Dataset and DataLoader objects.
- Split target data into training, validation, and test sets.

Supported dataset configurations:
- s2m: SVHN → MNIST
- u2m: USPS → MNIST
- m2u: MNIST → USPS
- m2mm: MNIST → MNIST-M
- 

### utils.py
This file provides shared utilities for experiment management.
It handles logging, metric aggregation, and reproducibility, enabling consistent monitoring and evaluation across experiments.
The utilities are independent of specific models or datasets, helping keep the main training code simple and reusable.

### Dataset Description
The experiments SVHN → MNIST (UDA) use the following datasets:
- [SVHN](http://ufldl.stanford.edu/housenumbers/)
- [MNIST](http://yann.lecun.com/exdb/mnist/)

#### SVHN
SVHN (Street View House Numbers) is a real-world digit recognition dataset collected from Google Street View images. It consists of color images of house number digits captured under varying lighting conditions, backgrounds, and viewpoints. Due to its natural image characteristics and visual complexity, SVHN is commonly used as a challenging source domain in domain adaptation experiments.

#### MNIST
MNIST is a benchmark handwritten digit dataset composed of grayscale images of digits from 0 to 9. The images are clean, centered, and have low visual variability compared to SVHN. In unsupervised domain adaptation, MNIST is often used as a target domain, representing a distribution shift from real-world images to simpler, handwritten digit data.

