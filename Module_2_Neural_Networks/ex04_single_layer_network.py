"""
Practical 2.4: Simple Single-Layer Neural Network (Perceptron Learning Rule)
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To create a simple single-layer neural network using Python.

Theory:
A Single-Layer Perceptron (Frank Rosenblatt, 1958) is capable of classifying linearly separable patterns.
The network model computes:
    z = w1*x1 + w2*x2 + b
    y_pred = 1 if z >= 0 else 0

Perceptron Learning Algorithm:
For each sample (x, y_true):
    Error e = y_true - y_pred
    w = w + η * e * x
    b = b + η * e
where η is the learning rate.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Perceptron:
    def __init__(self, lr=0.1, max_epochs=50):
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.max_epochs):
            total_errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                error = y[i] - y_pred

                if error != 0:
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error
                    total_errors += 1

            self.history.append((self.weights.copy(), self.bias, total_errors))
            if total_errors == 0:
                print(f"Perceptron converged at epoch {epoch + 1}!")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

def run_practical_2_4():
    print("=" * 60)
    print("PRACTICAL 2.4: Single-Layer Perceptron Implementation")
    print("=" * 60)

    # Linearly separable synthetic 2D dataset
    np.random.seed(42)
    # Class 0: Center (2, 2)
    X0 = np.random.randn(25, 2) * 0.6 + np.array([1.5, 1.5])
    y0 = np.zeros(25, dtype=int)
    # Class 1: Center (4, 4)
    X1 = np.random.randn(25, 2) * 0.6 + np.array([4.0, 4.0])
    y1 = np.ones(25, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # Train Perceptron
    clf = Perceptron(lr=0.05, max_epochs=30)
    clf.fit(X, y)

    print(f"\nTrained Weights: {clf.weights}")
    print(f"Trained Bias: {clf.bias:.4f}")

    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y) * 100.0
    print(f"Classification Accuracy: {accuracy:.2f}%\n")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Decision Boundary
    ax1.scatter(X0[:, 0], X0[:, 1], color='#e74c3c', s=50, label='Class 0', edgecolors='k')
    ax1.scatter(X1[:, 0], X1[:, 1], color='#2ecc71', s=50, label='Class 1', edgecolors='k')

    x_line = np.linspace(0, 6, 200)
    # w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
    if clf.weights[1] != 0:
        y_line = -(clf.weights[0] * x_line + clf.bias) / clf.weights[1]
        ax1.plot(x_line, y_line, 'k-', linewidth=2.2, label='Learned Decision Boundary')

    ax1.set_title("Single-Layer Perceptron Decision Boundary", fontweight='bold')
    ax1.set_xlabel("Feature x1")
    ax1.set_ylabel("Feature x2")
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Plot 2: Training Convergence (Errors vs Epoch)
    errors_per_epoch = [h[2] for h in clf.history]
    ax2.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, 'bo-', linewidth=2)
    ax2.set_title("Perceptron Learning Convergence", fontweight='bold')
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("Number of Misclassifications")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex04_single_layer_network.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.4 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_4()
