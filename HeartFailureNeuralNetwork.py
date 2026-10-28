"""
Heart Failure Prediction using a custom Neural Network implementation.

This script fetches the Heart Failure Clinical Records dataset, implements a 
Neural Network from scratch buuilt using NumPy, and performs 10-fold stratified 
cross-validation to evaluate different architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

def fetch_data():
    """Fetch the heart failure clinical records dataset."""
    heart_failure_clinical_records = fetch_ucirepo(id=519)
    X = heart_failure_clinical_records.data.features
    y = heart_failure_clinical_records.data.targets
    
    # Drop 'time' column as it's not a feature for prediction
    X = X.drop('time', axis=1)
    y = y.values.flatten()
    
    return X, y

def stratified_k_fold(X, y, n_splits=10, shuffle=True, seed=None):
    """
    Split data into stratified folds.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_splits: Number of folds
        shuffle: Whether to shuffle the indices
        seed: Random seed for reproducibility
        
    Yields:
        Tuple of (train_indices, val_indices)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    classes = np.unique(y)
    
    class_indices = {}
    for c in classes:
        indices = np.where(y == c)[0]
        if shuffle:
            rng.shuffle(indices)
        class_indices[c] = indices

    class_folds = {}
    for c in classes:
        indices = class_indices[c]
        n_class_samples = len(indices)
        fold_sizes = np.full(n_splits, n_class_samples // n_splits, dtype=int)
        fold_sizes[:n_class_samples % n_splits] += 1

        sofar = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = sofar, sofar + fold_size
            folds.append(indices[start:stop])
            sofar = stop
        class_folds[c] = folds

    for i in range(n_splits):
        val_indices = []
        train_indices = []
        for c in classes:
            val_indices.extend(class_folds[c][i])
            for j in range(n_splits):
                if j != i:
                    train_indices.extend(class_folds[c][j])

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        yield train_indices, val_indices

class NeuralNetwork:
    """A simple Neural Network implementation using NumPy."""
    
    def __init__(self, layer_sizes, learning_rate=0.01, batch_size=32, seed=None):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = self.rng.normal(0, np.sqrt(2.0 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -400, 400)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward_pass(self, X):
        self.activations = [X]
        self.preactivation_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.preactivation_values.append(z)

            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)

            self.activations.append(a)

        return self.activations[-1]

    def backward_pass(self, y_true):
        m = len(y_true)
        delta = self.activations[-1] - y_true

        gradients_w = []
        gradients_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.mean(delta, axis=0)

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.preactivation_values[i-1])

        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def create_batches(self, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))

        return batches

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, verbose=False):
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(epochs):
            batches = self.create_batches(X_train, y_train)
            for X_batch, y_batch in batches:
                y_pred = self.forward_pass(X_batch)
                gradients_w, gradients_b = self.backward_pass(y_batch)
                self.update_parameters(gradients_w, gradients_b)

            y_pred_train = self.forward_pass(X_train)
            train_loss = self.binary_cross_entropy(y_pred_train, y_train)
            train_acc = self.accuracy(y_pred_train, y_train)

            y_pred_val = self.forward_pass(X_val)
            val_loss = self.binary_cross_entropy(y_pred_val, y_val)
            val_acc = self.accuracy(y_pred_val, y_val)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        return train_loss_history, val_loss_history, train_acc_history, val_acc_history

    def binary_cross_entropy(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def accuracy(self, y_pred, y_true):
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions == y_true)

    def predict(self, X):
        return self.forward_pass(X)

    def get_predictions(self, X):
        y_pred = self.forward_pass(X)
        return (y_pred > 0.5).astype(int)

    def confusion_matrix(self, y_true, y_pred):
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        y_true_flat = y_true.flatten()

        tp = np.sum((y_pred_binary == 1) & (y_true_flat == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_flat == 0))
        fp = np.sum((y_pred_binary == 1) & (y_true_flat == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_flat == 1))

        return np.array([[tn, fp], [fn, tp]])

def main():
    # Fetch data
    X_raw, y_raw = fetch_data()
    
    # Configuration (change for more vigorous results)
    n_folds = 10
    epochs = 500
    learning_rate = 0.01
    batch_size = 32
    seed = 42
    
    architectures = {
        '2-Layer': [X_raw.shape[1], 32, 16, 1],
        '4-Layer': [X_raw.shape[1], 64, 32, 16, 8, 1],
        '6-Layer': [X_raw.shape[1], 128, 64, 32, 16, 8, 1]
    }

    print("=" * 70)
    print("10-FOLD CROSS-VALIDATION (Batch-Gradient Descent)")
    print(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
    print(f"Batch Size: {batch_size}")
    print("=" * 70)

    cv_results = {}
    for arch_name, architecture in architectures.items():
        print(f"\n{arch_name} Network: {architecture}")
        print("-" * 70)

        fold_results = {
            'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
            'train_loss_history': [], 'val_loss_history': [],
            'confusion_matrices': [], 'models': []
        }

        fold_num = 1
        for train_idx, val_idx in stratified_k_fold(X_raw, y_raw, n_splits=n_folds, shuffle=True, seed=seed):
            print(f"  Fold {fold_num}/{n_folds}...", end=" ")

            X_train, X_val = X_raw[train_idx], X_raw[val_idx]
            y_train, y_val = y_raw[train_idx].reshape(-1, 1), y_raw[val_idx].reshape(-1, 1)

            # Normalise using training set statistics
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            std[std == 0] = 1.0

            X_train_norm = (X_train - mean) / std
            X_val_norm = (X_val - mean) / std

            nn = NeuralNetwork(architecture, learning_rate=learning_rate,
                              batch_size=batch_size, seed=seed + fold_num)
            
            train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = nn.train(
                X_train_norm, y_train, X_val_norm, y_val,
                epochs=epochs, verbose=False
            )

            fold_results['train_acc'].append(train_acc_hist[-1])
            fold_results['val_acc'].append(val_acc_hist[-1])
            fold_results['train_loss'].append(train_loss_hist[-1])
            fold_results['val_loss'].append(val_loss_hist[-1])
            fold_results['train_loss_history'].append(train_loss_hist)
            fold_results['val_loss_history'].append(val_loss_hist)

            y_pred_val = nn.forward_pass(X_val_norm)
            cm = nn.confusion_matrix(y_val, y_pred_val)
            fold_results['confusion_matrices'].append(cm)
            fold_results['models'].append(nn)

            print(f"Train Acc: {train_acc_hist[-1]:.4f}, Val Acc: {val_acc_hist[-1]:.4f}")
            fold_num += 1

        cv_results[arch_name] = {
            'mean_train_acc': np.mean(fold_results['train_acc']),
            'std_train_acc': np.std(fold_results['train_acc']),
            'mean_val_acc': np.mean(fold_results['val_acc']),
            'std_val_acc': np.std(fold_results['val_acc']),
            'mean_train_loss': np.mean(fold_results['train_loss']),
            'std_train_loss': np.std(fold_results['train_loss']),
            'mean_val_loss': np.mean(fold_results['val_loss']),
            'std_val_loss': np.std(fold_results['val_loss']),
            'fold_results': fold_results
        }

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for arch_name, stats in cv_results.items():
        print(f"\nArchitecture: {arch_name}")
        print(f"  Mean Validation Accuracy: {stats['mean_val_acc']:.4f} \u00B1 {stats['std_val_acc']:.4f}")
        print(f"  Mean Training Accuracy: {stats['mean_train_acc']:.4f} \u00B1 {stats['std_train_acc']:.4f}")
        print(f"  Mean Validation Loss: {stats['mean_val_loss']:.4f} \u00B1 {stats['std_val_loss']:.4f}")

if __name__ == "__main__":
    main()
