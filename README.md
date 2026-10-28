# Heart Failure Prediction - Neural Network from Scratch

This project implements a custom Neural Network using only NumPy to predict heart failure based on clinical records. It features a full implementation of forward and backward passes, mini-batch gradient descent, and a 10-fold stratified cross-validation process.

## Features
- **Custom Neural Network**: Implemented from scratch using NumPy (no high-level ML libraries like Keras or PyTorch).
- **Mini-Batch Gradient Descent**: Efficiently updates weights using mini-batches.
- **Stratified K-Fold Cross-Validation**: Ensures robust evaluation by maintaining class proportions across folds.
- **Multiple Architectures**: Evaluates 2-layer, 4-layer, and 6-layer networks.
- **Metrics**: Tracks training/validation loss and accuracy across epochs.

## Dataset
The project uses the **Heart Failure Clinical Records** dataset from the UCI Machine Learning Repository.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MLCOURSEWORK2
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the evaluation:
```bash
python MLCOURSEWORK2CODE.py
```

## Architecture Details
- **Activation Functions**: ReLU for hidden layers, Sigmoid for the output layer.
- **Weight Initialization**: He initialization (scaled normal distribution).
- **Optimization**: Standard Gradient Descent with a configurable learning rate and batch size.

## Results
The script outputs a summary of the mean validation and training accuracy for three different network architectures, providing a clear comparison of model performance.
