"""
Practical 2.2: Weighted Sum of Inputs and Bias
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To implement the weighted sum of inputs and bias used in an artificial neuron using Python.

Theory:
In modern neural architectures, each neuron calculates an affine transformation:
    z = w^T · x + b = (∑_{i=1}^n w_i * x_i) + b

Where:
- x = [x1, x2, ..., xn]^T is the input feature vector.
- w = [w1, w2, ..., wn]^T is the synaptic weight vector defining orientation of the hyperplane.
- b is the scalar bias, acting as an adjustable intercept that allows the decision boundary
  to shift freely without being constrained to pass through the origin.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_affine_transformation(X, W, b):
    """
    Vectorized computation of weighted sum + bias:
    X: shape (m, n) where m is batch size, n is feature dimension
    W: shape (n, 1)
    b: scalar or shape (1,)
    Returns Z = X @ W + b
    """
    return np.dot(X, W) + b

def run_practical_2_2():
    print("=" * 60)
    print("PRACTICAL 2.2: Weighted Sum and Bias Implementation")
    print("=" * 60)

    # 1. Concrete Numerical Demonstration
    X_sample = np.array([
        [1.5, 2.0],
        [3.0, 4.5],
        [-1.0, 2.5],
        [0.0, 1.0]
    ])
    W = np.array([[0.8], [-1.2]])
    bias = 0.5

    Z = compute_affine_transformation(X_sample, W, bias)

    print(f"Weights W:\n{W.ravel()}")
    print(f"Bias b: {bias}")
    print("\nInput Vector X and Computed Linear Activation Z = W^T*X + b:")
    print(f"{'Sample':^8} | {'x1':^8} | {'x2':^8} | {'Weighted Sum (z)':^18}")
    print("-" * 50)
    for i in range(len(X_sample)):
        print(f"{i+1:^8} | {X_sample[i, 0]:8.2f} | {X_sample[i, 1]:8.2f} | {Z[i, 0]:18.3f}")

    # 2. Geometric Effect of Bias Visualization
    # Hyperplane equation: w1*x1 + w2*x2 + b = 0 => x2 = -(w1/w2)*x1 - (b/w2)
    x1_vals = np.linspace(-5, 5, 200)
    w1, w2 = 1.5, -2.0
    biases = [-4.0, -2.0, 0.0, 2.0, 4.0]

    plt.figure(figsize=(10, 6))
    colors = ['#c0392b', '#e67e22', '#2c3e50', '#27ae60', '#2980b9']

    for b_val, c in zip(biases, colors):
        x2_vals = -(w1 / w2) * x1_vals - (b_val / w2)
        style = '--' if b_val == 0.0 else '-'
        label = f'Bias b = {b_val:+.1f}' if b_val != 0.0 else 'Bias b = 0.0 (Through Origin)'
        plt.plot(x1_vals, x2_vals, color=c, linestyle=style, linewidth=2.2, label=label)

    plt.title(f"Geometric Shift of Separating Hyperplane ({w1}*x1 + {w2}*x2 + b = 0) with Varying Bias",
              fontsize=12, fontweight='bold')
    plt.xlabel("Feature x1", fontsize=11)
    plt.ylabel("Feature x2", fontsize=11)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', framealpha=0.95)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex02_weighted_sum_and_bias.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.2 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_2()
