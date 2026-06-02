# Brain Tumor MRI Classification Using CNNs

A deep learning project that classifies brain MRI scans into four categories — **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor** — using Convolutional Neural Networks built with PyTorch.

## Overview

This project compares three model architectures of increasing depth and complexity to evaluate the effectiveness of CNNs for medical image classification:

- **Baseline MLP** — flattens images into pixel vectors; achieved **77.69% accuracy**
- **CNN Model 1** — two convolutional layers with max pooling; achieved **87.38% accuracy**
- **CNN Model 2** — four convolutional layers with six fully connected layers; achieved **89.62% accuracy**

The results confirm that deeper CNN architectures with spatial feature extraction significantly outperform pixel-level classification for medical imaging tasks.

## Dataset

- ~7,000 preprocessed MRI images in JPEG format sourced from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- 4 classes: Glioma, Meningioma, Pituitary, No Tumor
- Images resized to 150x150 pixels and converted to grayscale

## Tech Stack

- Python
- PyTorch
- scikit-learn
- matplotlib
- kagglehub

## Files

- `MRI_Project.ipynb` — full notebook with code, training, and evaluation
- `MRI_Project.py` — standalone Python script
- `Report.pdf` — full written report with methodology, results, and analysis

## Results Summary

| Model     | Accuracy | Macro F1 |
|-----------|----------|----------|
| MLP       | 77.69%   | 0.777    |
| CNN Model 1 | 87.38% | 0.874    |
| CNN Model 2 | 89.62% | 0.896    |

Glioma was consistently the most challenging class to classify across all models, likely due to its visual similarity to other tumor types.

## Future Work

- Implement saliency maps to visualize which regions of the MRI the model focuses on
- Explore transfer learning with pretrained architectures (e.g. ResNet, VGG)
