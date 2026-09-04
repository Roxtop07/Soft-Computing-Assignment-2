"""
Practical 2.1: Basic Working of an Artificial Neuron
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To understand and implement the basic working of an artificial neuron using Python.

Theory:
Proposed by Warren McCulloch and Walter Pitts (1943), the artificial neuron is the fundamental
computational unit of ANNs. It mimics the biological neuron:
- Dendrites receive input signals (x1, x2, ..., xn)
- Synapses scale signals by weights (w1, w2, ..., wn)
- Cell body (Soma) aggregates the inputs and subtracts a threshold θ:
      Net Input = ∑_{i=1}^n w_i * x_i
- Axon outputs firing decision based on a threshold activation function:
      y = 1 if Net Input >= θ else 0
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = np.array(weights)
        self.threshold = threshold

    def activate(self, inputs):
        net = np.dot(inputs, self.weights)
        return 1 if net >= self.threshold else 0

def run_practical_2_1():
    print("=" * 60)
    print("PRACTICAL 2.1: McCulloch-Pitts Artificial Neuron")
    print("=" * 60)

    # 1. Implementation of AND Gate: inputs x1, x2 ∈ {0, 1}
    # Weights = [1, 1], Threshold θ = 2
    and_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=2)

    # 2. Implementation of OR Gate: inputs x1, x2 ∈ {0, 1}
    # Weights = [1, 1], Threshold θ = 1
    or_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=1)

    # 3. Implementation of NOT Gate: input x1 ∈ {0, 1}
    # Weight = -1, Threshold θ = 0
    not_neuron = McCullochPittsNeuron(weights=[-1], threshold=0)

    inputs_2d = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("\nTruth Table Verification:")
    print(f"{'x1':^5} | {'x2':^5} | {'AND Output':^12} | {'OR Output':^12}")
    print("-" * 42)
    and_outs, or_outs = [], []
    for x1, x2 in inputs_2d:
        y_and = and_neuron.activate([x1, x2])
        y_or = or_neuron.activate([x1, x2])
        and_outs.append(y_and)
        or_outs.append(y_or)
        print(f"{x1:^5} | {x2:^5} | {y_and:^12} | {y_or:^12}")

    print("\nNOT Gate Verification:")
    print(f"{'x1':^5} | {'NOT Output':^12}")
    print("-" * 20)
    for x in [0, 1]:
        print(f"{x:^5} | {not_neuron.activate([x]):^12}")

    # Visualization of Decision Boundaries for AND and OR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # AND Decision Boundary: w1*x1 + w2*x2 = theta -> x1 + x2 = 2 -> x2 = 2 - x1
    x_line = np.linspace(-0.2, 1.2, 100)
    ax1.plot(x_line, 2 - x_line, 'k--', label='Boundary: x1 + x2 = 2')
    for (x1, x2), y in zip(inputs_2d, and_outs):
        marker = 'o' if y == 1 else 'x'
        color = 'green' if y == 1 else 'red'
        ax1.scatter(x1, x2, color=color, s=120, marker=marker, linewidths=2.5,
                    label=f'Class {y}' if f'Class {y}' not in ax1.get_legend_handles_labels()[1] else '')
    ax1.set_title("AND Gate Decision Boundary (θ = 2)", fontweight='bold')
    ax1.set_xlabel("Input x1")
    ax1.set_ylabel("Input x2")
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # OR Decision Boundary: x1 + x2 = 1 -> x2 = 1 - x1
    ax2.plot(x_line, 1 - x_line, 'k--', label='Boundary: x1 + x2 = 1')
    for (x1, x2), y in zip(inputs_2d, or_outs):
        marker = 'o' if y == 1 else 'x'
        color = 'green' if y == 1 else 'red'
        ax2.scatter(x1, x2, color=color, s=120, marker=marker, linewidths=2.5,
                    label=f'Class {y}' if f'Class {y}' not in ax2.get_legend_handles_labels()[1] else '')
    ax2.set_title("OR Gate Decision Boundary (θ = 1)", fontweight='bold')
    ax2.set_xlabel("Input x1")
    ax2.set_ylabel("Input x2")
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex01_artificial_neuron.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.1 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_1()
