# Iris Recognition System

## Project Overview

This project implements an Iris Recognition System using MATLAB and Machine Learning techniques for biometric identification.

The system extracts iris features using HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns), reduces feature dimensions using PCA, and compares multiple classifiers for recognition.

---

## Technologies Used

- MATLAB
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

---

## Dataset

**MMU Iris Database**

The project uses the publicly available MMU Iris Database for training and testing.

---

## Algorithms Used

### Image Preprocessing

- Median Filter
- Gaussian Filter
- Adaptive Histogram Equalization

### Feature Extraction

- HOG
- LBP

### Feature Reduction

- Principal Component Analysis (PCA)

### Classification

- Support Vector Machine (SVM)
- K-Nearest Neighbor (KNN)
- Random Forest

---

## Project Workflow

Input Iris Image

↓

Image Preprocessing

↓

Feature Extraction (HOG + LBP)

↓

Feature Fusion

↓

PCA

↓

Model Training

↓

Prediction

---

## Project Files

| File | Description |
|------|-------------|
| 01_FeatureExtraction.m | Extracts HOG and LBP features |
| 02_ModelTraining.m | Trains SVM, KNN and Random Forest |
| 03_TestPrediction.m | Predicts identity for a new iris image |

---

## Author

Aditya Pawar
