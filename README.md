# Brain Tumor Segmentation using U-Net and Swin Transformer

## Overview
This repository contains a Jupyter Notebook for brain tumor segmentation using deep learning models, specifically U-Net. The project is based on the BraTS 2020 dataset and aims to compare the performance of CNN-based and Transformer-based architectures for multi-class tumor segmentation. 

## Present Work
The current work focuses on implementing U-Net model for brain tumor segmentation as a baseline for CNN based models. The model is trained on the BraTS 2020 dataset, which consists of four MRI sequences (T1, T1Gd, T2, FLAIR) and three classes (necrotic tumor core, peritumoral edema, enhancing tumor core). The models are evaluated based on the Dice coefficient and Intersection over Union (IoU) metrics. 

## Future Work
The future work will involve implementing Swin Transformer model for brain tumor segmentation and comparing its performance with U-Net. The Swin Transformer is a transformer-based architecture that has shown promising results in various computer vision tasks. The goal is to explore the potential of transformer-based models in medical image segmentation tasks and evaluate their performance on the BraTS dataset.
- Improving transformer-based models using synthetic data augmentation techniques.
- Exploring diffusion models for better synthetic data generation.
- Optimizing computational efficiency in segmentation models.


## Dataset
- **Name**: BraTS 2020 (Brain Tumor Segmentation Challenge Dataset)
- **Modalities**: Four MRI sequences (T1, T1Gd, T2, FLAIR)
- **Classes**:
  - Background
  - Necrotic tumor core
  - Peritumoral edema
  - Enhancing tumor core

## Methodology
1. **Data Preprocessing**:
   - Normalization of MRI scans
   - Data augmentation using Albumentations
   - Splitting into training, validation, and test sets
2. **Model Selection**:
   - **U-Net**: A convolutional neural network (CNN)-based architecture
   - **Swin Transformer**: A transformer-based architecture for segmentation
3. **Training and Evaluation**:
   - Loss function: Dice Loss & Cross Entropy
   - Metrics: Dice coefficient, IoU
   - Optimizer: Adam or SGD
   - Learning rate scheduling
   
## Results and Observations
- **U-Net** achieved a Dice coefficient of ~0.6.
- Challenges faced: Class imbalance


## Dependencies
- Python 3.x
- PyTorch
- Albumentations
- MONAI (for medical imaging preprocessing)
- scikit-learn
- matplotlib, numpy, pandas

## Usage
Run the Jupyter Notebook to train and evaluate the models. Adjust hyperparameters as needed to fine-tune performance.

## Contact
For further inquiries, feel free to reach out.
