"""
Practical 2.3: Activation Functions and Their Derivatives
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To implement and visualize Sigmoid, Tanh, and ReLU activation functions using Python.

Theory:
Non-linear activation functions allow multilayer neural networks to learn complex non-linear mappings:
1. Sigmoid (Logistic):
       σ(z) = 1 / (1 + e^(-z)) ∈ (0, 1)
       Derivative: σ'(z) = σ(z) * (1 - σ(z))
2. Hyperbolic Tangent (Tanh):
       tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)) ∈ (-1, 1)
       Derivative: tanh'(z) = 1 - tanh^2(z)
3. Rectified Linear Unit (ReLU):
       f(z) = max(0, z)
       Derivative: f'(z) = 1 if z > 0 else 0
4. Leaky ReLU:
       f(z) = z if z > 0 else α*z (α = 0.01)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

# 2. Tanh
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1.0 - np.tanh(z)**2

# 3. ReLU
def relu(z):
    return np.maximum(0.0, z)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

# 4. Leaky ReLU
def leaky_relu(z, alpha=0.1):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.1):
    return np.where(z > 0, 1.0, alpha)

def run_practical_2_3():
    print("=" * 60)
    print("PRACTICAL 2.3: Activation Functions and Derivatives")
    print("=" * 60)

    z = np.linspace(-6, 6, 600)

    # Numerical outputs at critical test points
    test_pts = [-3.0, -1.0, 0.0, 1.0, 3.0]
    print(f"{'z':^6} | {'Sigmoid':^10} | {'Tanh':^10} | {'ReLU':^10} | {'dSigmoid':^10} | {'dTanh':^10}")
    print("-" * 65)
    for p in test_pts:
        print(f"{p:^6.1f} | {sigmoid(p):^10.4f} | {tanh(p):^10.4f} | {relu(p):^10.4f} | {sigmoid_derivative(p):^10.4f} | {tanh_derivative(p):^10.4f}")

    # Visualization: 2x2 grid (Functions and Derivatives)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-Left: Activation Functions
    axes[0, 0].plot(z, sigmoid(z), 'b-', linewidth=2.2, label='Sigmoid')
    axes[0, 0].plot(z, tanh(z), 'g-', linewidth=2.2, label='Tanh')
    axes[0, 0].plot(z, relu(z), 'r-', linewidth=2.2, label='ReLU')
    axes[0, 0].plot(z, leaky_relu(z), 'm--', linewidth=2.0, label='Leaky ReLU (α=0.1)')
    axes[0, 0].set_title("Activation Functions f(z)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(-1.5, 3.5)
    axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.7)
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend(loc='upper left')

    # Top-Right: Derivatives f'(z)
    axes[0, 1].plot(z, sigmoid_derivative(z), 'b-', linewidth=2.2, label="Sigmoid' (Peak = 0.25)")
    axes[0, 1].plot(z, tanh_derivative(z), 'g-', linewidth=2.2, label="Tanh' (Peak = 1.0)")
    axes[0, 1].plot(z, relu_derivative(z), 'r-', linewidth=2.2, label="ReLU'")
    axes[0, 1].plot(z, leaky_relu_derivative(z), 'm--', linewidth=2.0, label="Leaky ReLU'")
    axes[0, 1].set_title("Derivatives of Activation Functions f'(z)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(-0.1, 1.2)
    axes[0, 1].axhline(0, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].axvline(0, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend(loc='upper right')

    # Bottom-Left: Sigmoid Saturation & Vanishing Gradient demonstration
    axes[1, 0].plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
    axes[1, 0].fill_between(z[z < -3], sigmoid(z[z < -3]), color='red', alpha=0.2, label='Saturation Regions (f\' ≈ 0)')
    axes[1, 0].fill_between(z[z > 3], sigmoid(z[z > 3]), color='red', alpha=0.2)
    axes[1, 0].set_title("Sigmoid Saturation & Vanishing Gradient Zones", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Input z", fontsize=11)
    axes[1, 0].set_ylabel("Activation σ(z)", fontsize=11)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend(loc='upper left')

    # Bottom-Right: ReLU vs Leaky ReLU (Dying ReLU mitigation)
    axes[1, 1].plot(z, relu(z), 'r-', linewidth=2.2, label='Standard ReLU (zero for z<0)')
    axes[1, 1].plot(z, leaky_relu(z, 0.1), 'm--', linewidth=2.2, label='Leaky ReLU (leak = 0.1*z)')
    axes[1, 1].set_title("Dying ReLU Solution: Leaky ReLU vs. Standard ReLU", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Input z", fontsize=11)
    axes[1, 1].set_ylabel("f(z)", fontsize=11)
    axes[1, 1].set_ylim(-1.0, 3.5)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend(loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex03_activation_functions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.3 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_3()
