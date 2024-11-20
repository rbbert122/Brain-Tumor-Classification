# Brain-Tumor-Classification

## Overview

This project focuses on developing a machine learning model for automatic brain tumor classification using medical imaging. The goal is to classify brain tumor images into different categories with high accuracy and reliability.

## Dataset Characteristics

- **Total Images**: 3,612 medical images
- **Image Format**: RGB
- **Predominant Image Size**: 512×512 pixels
- **Pixel Depth**: 8-bit (0-255 range)
- **Image Preprocessing**: Resized to 227×227 pixels for model consistency

## Classes

The project classifies brain tumors into four categories:

- Glioma Tumors
- Meningioma Tumors
- Pituitary Tumors
- No Tumor

## Key Challenges

- Significant intra-class variability in tumor appearances
- Inter-class similarities that complicate classification
- Class imbalance in the original dataset

## Methodology

### Data Preprocessing

- Random oversampling to address class imbalance
- Data augmentation techniques:
  - Vertical flipping
  - Random rotation (up to 45 degrees)
  - Gaussian blur
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Sobel edge detection
  - Normalization

### Model Architecture

- **Network**: AlexNet
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Stochastic Gradient Descent
- **Training Duration**: 90 epochs

### Hyperparameters

- Learning Rate: 0.01
- Momentum: 0.9
- Weight Decay: 0.0005
- Learning Rate Scheduler: 10-fold decrease with 10-epoch patience

## Performance Metrics

- **Test Accuracy**: 73.24%
- **F1 Score**: 65.23%
- **Precision**: 65.23%
- **Recall**: 65.23%

## Visualization

Detailed visualizations include:

- Class distribution across train/validation/test sets
- Sample images from each tumor class
- Model training and validation metrics
- Confusion matrix showing prediction performance

## Key Insights

- Significant variability within tumor classes
- Some overlap between tumor type characteristics
- Importance of diverse and representative training data
