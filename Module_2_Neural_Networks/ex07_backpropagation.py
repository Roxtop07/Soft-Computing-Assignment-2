"""
Practical 2.7: Backpropagation Algorithm from Scratch
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To understand and implement the basic concept of Backpropagation for training a neural network.

Theory:
Backpropagation (Rumelhart, Hinton, & Williams, 1986) calculates the gradient of the loss function
with respect to each weight using the Chain Rule of calculus:
1. Forward Pass:
       z1 = W1 * x + b1,   a1 = σ(z1)
       z2 = W2 * a1 + b2,  a2 = σ(z2) (y_hat)
       Loss L = 0.5 * (y - y_hat)^2
2. Backward Pass (Output Layer):
       δ2 = ∂L / ∂z2 = (y_hat - y) * σ'(z2)
       ∂L / ∂W2 = a1^T * δ2,   ∂L / ∂b2 = δ2
3. Backward Pass (Hidden Layer):
       δ1 = ∂L / ∂z1 = (δ2 * W2^T) * σ'(z1)
       ∂L / ∂W1 = x^T * δ1,    ∂L / ∂b1 = δ1
4. Weight Updates:
       W = W - η * (∂L / ∂W)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)

class BackpropagationNN:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, lr=0.5):
        self.lr = lr
        np.random.seed(42)
        # Initialize weights with Xavier/Glorot scaling
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.loss_history = []

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        # Output layer error
        delta2 = (y_pred - y) * sigmoid_prime(self.z2)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # Hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_prime(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # Parameter update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Mean Squared Error Loss
        mse = 0.5 * np.mean((y - y_pred) ** 2)
        return mse

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.backward(X, y, y_pred)
            self.loss_history.append(loss)
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1:4d} | MSE Loss = {loss:.6f}")

def run_practical_2_7():
    print("=" * 60)
    print("PRACTICAL 2.7: Backpropagation Algorithm from Scratch")
    print("=" * 60)

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = BackpropagationNN(input_dim=2, hidden_dim=4, output_dim=1, lr=1.5)
    nn.train(X, y, epochs=1200)

    final_preds = nn.forward(X)
    print("\nTrained Network Predictions on XOR:")
    print(f"{'x1':^5} | {'x2':^5} | {'y_true':^8} | {'y_pred':^12} | {'Class':^8}")
    print("-" * 46)
    for i in range(4):
        p = float(final_preds[i, 0])
        print(f"{X[i, 0]:^5} | {X[i, 1]:^5} | {y[i, 0]:^8} | {p:^12.4f} | {round(p):^8}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Loss Curve
    ax1.plot(range(1, len(nn.loss_history) + 1), nn.loss_history, 'crimson', linewidth=2)
    ax1.set_title("Backpropagation MSE Loss Convergence Curve", fontweight='bold')
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Decision Surface
    x_span = np.linspace(-0.2, 1.2, 100)
    y_span = np.linspace(-0.2, 1.2, 100)
    XX, YY = np.meshgrid(x_span, y_span)
    grid = np.c_[XX.ravel(), YY.ravel()]
    ZZ = nn.forward(grid).reshape(XX.shape)

    contour = ax2.contourf(XX, YY, ZZ, levels=20, cmap='Spectral', alpha=0.8)
    fig.colorbar(contour, ax=ax2, label='Predicted Output')
    ax2.scatter([0, 1], [0, 1], c='blue', s=120, edgecolors='k', label='Class 0')
    ax2.scatter([0, 1], [1, 0], c='orange', s=120, edgecolors='k', label='Class 1')
    ax2.set_title("Learned Decision Surface via Backpropagation", fontweight='bold')
    ax2.set_xlabel("Input x1")
    ax2.set_ylabel("Input x2")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex07_backpropagation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.7 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_7()
