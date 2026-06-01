># Speech / Audio Classification System (KNN & MLP)
- [1. Introduction](#1-introduction)
- [2. Architecture](#2-architecture)
- [3. Implementation Overview](#3-implementation-overview)
  - [Tools Used](#tools-used)
  - [Dataset Overview](#dataset-overview)
- [4. Code Structure](#4-code-structure)
- [5. Feature Extraction](#5-feature-extraction)
- [6. Dataset Builder](#6-dataset-builder)
- [7. KNN Model](#7-knn-model)
  - [Overview](#overview)
  - [Training Steps](#training-steps)
- [8. MLP Model](#8-mlp-model)
  - [Overview](#overview-1)
  - [Architecture](#architecture)
  - [Training Process](#training-process)
  - [Why One-Hot Encoding?](#why-one-hot-encoding)
  - [Training Configuration](#training-configuration)
- [9. Model Inference (`testModel`)](#9-model-inference-testmodel)
- [10. Transcript Evaluation](#10-transcript-evaluation)
  - [Process](#process)
  - [Metrics](#metrics)
- [11. Execution](#11-execution)


---

## 1. Introduction

This project implements a Speech Activity Detection (SAD) system that classifies audio signals into:

- foreground (speech)
- background (noise)

Two machine learning approaches are used:

- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)

Trained models are saved in `.mat` format inside the `models/` directory.

---

## 2. Architecture

The system follows a modular processing pipeline:

- Audio loading and preprocessing
- Segmentation into fixed-length chunks
- Feature extraction per chunk
- Model training (KNN / MLP)
- Prediction on unseen audio
- Post-processing (segment merging)
- CSV export of results

---

## 3. Implementation Overview

### Tools Used
- MATLAB R2025b
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

### Dataset Overview
- Noise files: multiple recordings
- Speech files: dataset-based recordings
- Balanced dynamically during training

---

## 4. Code Structure

Main modules:

- `dataLoader` → loads dataset paths
- `extractFeatures` → feature extraction
- `datasetBuilder` → dataset creation
- `trainKnnModel` → KNN training
- `trainMLPModel` → neural network training
- `testModel` → inference pipeline
- `compareWithTranscript` → evaluation against ground truth

---

## 5. Feature Extraction

Each audio chunk is transformed into a feature vector:

- MFCC (13 coefficients)
- Energy (signal power)
- Zero Crossing Rate (signal variability)
- Frequency-domain features using `freqz`
  - Mean magnitude
  - Standard deviation

These features represent both temporal and spectral properties of audio signals.

---

## 6. Dataset Builder

Process:

- Load audio files
- Convert to mono signal
- Split into fixed-size chunks
- Extract features per chunk
- Assign labels (foreground / background)
- Shuffle dataset for training stability

---

## 7. KNN Model

### Overview
KNN classifies samples based on similarity in feature space.

### Training Steps
- Feature extraction
- Dataset balancing (speech vs noise)
- Random shuffling
- Training using:

```matlab
fitcknn(X, Y, ...
    NumNeighbors=3, ...
    Distance="euclidean", ...
    Standardize=true)
```
## 8. MLP Model
### Overview
---
The Multi-Layer Perceptron learns nonlinear relationships between audio features and class labels.

### Architecture
---
Input layer: feature vector
Hidden layer 1: 64 neurons
Hidden layer 2: 32 neurons
Output layer: 2 classes (foreground / background)
### Training Process
Convert labels to categorical values
Apply one-hot encoding to labels
Define neural network architecture using patternnet
Train using Levenberg–Marquardt optimization
### Why One-Hot Encoding?

One-hot encoding is used because:

* Neural networks require numeric vector targets
* It prevents ordinal interpretation of classes
* Each class is represented independently

Example:

foreground → [1 0]
background → [0 1]

### Training Configuration
* Training function: `trainlm`
* Epochs: 500
* Early stopping: validation-based (max_fail)
* Data split:
  * 70% training
  * 15% validation
  * 15% testing
## 9. Model Inference (`testModel`)

Pipeline:

* Load audio file
* Convert to mono
* Apply sliding window segmentation
* Extract features per chunk
* Predict:
* KNN → nearest neighbor classification
* MLP → probability-based classification
* Apply smoothing filter (majority vote)
* Merge consecutive segments
* Export results to CSV
## 10. Transcript Evaluation
### Process
1. Load JSON transcript
2. Convert timestamps to seconds
3. Assign labels using keyword rules
4. Compare predicted segments with ground truth
5. Use Intersection over Union (IoU)

A prediction is considered correct if:
* IoU > 0.5
---
### Metrics
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
## 11. Execution

Run the full pipeline:

`mainTrainModel`

Configuration is controlled via the `config` struct.