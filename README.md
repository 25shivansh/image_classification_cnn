# Image Classification using CNN

A Convolutional Neural Network built with PyTorch to classify images from the CIFAR-10 dataset.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32x32 color images across 10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

- Training samples: 50,000
- Test samples: 10,000

## Model Architecture

3 convolutional blocks followed by fully connected layers:

- **Conv Block 1:** Conv2d(3→32) → ReLU → MaxPool
- **Conv Block 2:** Conv2d(32→64) → ReLU → MaxPool
- **Conv Block 3:** Conv2d(64→128) → ReLU → MaxPool
- **FC Layers:** Linear(2048→256) → ReLU → Linear(256→10)

## Training

| Parameter   | Value              |
|-------------|--------------------|
| Epochs      | 10                 |
| Batch Size  | 64                 |
| Optimizer   | Adam               |
| Loss        | CrossEntropyLoss   |

Training loss dropped from **1.39** (epoch 1) to **0.18** (epoch 10).

## Result

**Test Accuracy: 75.02%**

## Requirements

```bash
pip install torch torchvision
```

## Usage

Open and run `cnn.ipynb` in Jupyter Notebook or JupyterLab.
The CIFAR-10 dataset will be downloaded automatically on first run.

