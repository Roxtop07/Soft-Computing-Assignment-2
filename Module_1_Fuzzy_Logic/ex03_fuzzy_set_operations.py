"""
Practical 1.3: Fuzzy Set Operations (Union, Intersection, Complement)
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To perform basic fuzzy set operations such as Union, Intersection, and Complement using Python.

Theory:
For two fuzzy sets A and B defined over universe X with membership functions μ_A(x) and μ_B(x):
1. Union (S-norm / T-conorm):
   μ_(A ∪ B)(x) = max(μ_A(x), μ_B(x))
2. Intersection (T-norm):
   μ_(A ∩ B)(x) = min(μ_A(x), μ_B(x))
3. Complement (Negation):
   μ_(~A)(x) = 1 - μ_A(x)
4. Algebraic Product:
   μ_(A · B)(x) = μ_A(x) * μ_B(x)
5. Algebraic Sum:
   μ_(A ⊕ B)(x) = μ_A(x) + μ_B(x) - μ_A(x) * μ_B(x)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a), (c - x)/(c - b)), 0)

def run_practical_3():
    print("=" * 60)
    print("PRACTICAL 1.3: Fuzzy Set Operations (Union, Intersection, Complement)")
    print("=" * 60)

    x = np.linspace(0, 10, 500)

    # Define two overlapping fuzzy sets
    # Set A: "Approximately 3"
    mu_A = trimf(x, 1.0, 3.0, 5.0)
    # Set B: "Approximately 5"
    mu_B = trimf(x, 3.0, 5.0, 8.0)

    # Standard Zadeh Operations
    union_AB = np.maximum(mu_A, mu_B)
    intersect_AB = np.minimum(mu_A, mu_B)
    comp_A = 1.0 - mu_A

    # Additional standard algebraic operators
    alg_product = mu_A * mu_B
    alg_sum = mu_A + mu_B - (mu_A * mu_B)

    # Sample evaluations
    sample_indices = [100, 200, 300]
    print(f"{'x':>6} | {'mu_A':>8} | {'mu_B':>8} | {'Union':>8} | {'Inter':>8} | {'Comp_A':>8}")
    print("-" * 56)
    for idx in sample_indices:
        print(f"{x[idx]:6.2f} | {mu_A[idx]:8.3f} | {mu_B[idx]:8.3f} | {union_AB[idx]:8.3f} | {intersect_AB[idx]:8.3f} | {comp_A[idx]:8.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Plot 1: Base Fuzzy Sets A and B
    axes[0, 0].plot(x, mu_A, 'b-', linewidth=2.2, label='Fuzzy Set A')
    axes[0, 0].plot(x, mu_B, 'r-', linewidth=2.2, label='Fuzzy Set B')
    axes[0, 0].set_title("Input Fuzzy Sets A and B", fontweight='bold')
    axes[0, 0].set_ylabel("Membership Degree μ(x)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # Plot 2: Union (Max)
    axes[0, 1].plot(x, mu_A, 'b--', alpha=0.4, label='Set A')
    axes[0, 1].plot(x, mu_B, 'r--', alpha=0.4, label='Set B')
    axes[0, 1].plot(x, union_AB, 'purple', linewidth=2.5, label='Union: max(A, B)')
    axes[0, 1].fill_between(x, union_AB, color='purple', alpha=0.15)
    axes[0, 1].set_title("Fuzzy Union: A ∪ B", fontweight='bold')
    axes[0, 1].set_ylabel("Membership Degree μ(x)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()

    # Plot 3: Intersection (Min)
    axes[1, 0].plot(x, mu_A, 'b--', alpha=0.4, label='Set A')
    axes[1, 0].plot(x, mu_B, 'r--', alpha=0.4, label='Set B')
    axes[1, 0].plot(x, intersect_AB, 'green', linewidth=2.5, label='Intersection: min(A, B)')
    axes[1, 0].fill_between(x, intersect_AB, color='green', alpha=0.25)
    axes[1, 0].set_title("Fuzzy Intersection: A ∩ B", fontweight='bold')
    axes[1, 0].set_xlabel("Universe X")
    axes[1, 0].set_ylabel("Membership Degree μ(x)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend()

    # Plot 4: Complement of A
    axes[1, 1].plot(x, mu_A, 'b--', alpha=0.5, label='Set A')
    axes[1, 1].plot(x, comp_A, 'orange', linewidth=2.5, label='Complement: 1 - A')
    axes[1, 1].fill_between(x, comp_A, color='orange', alpha=0.15)
    axes[1, 1].set_title("Fuzzy Complement: ~A", fontweight='bold')
    axes[1, 1].set_xlabel("Universe X")
    axes[1, 1].set_ylabel("Membership Degree μ(x)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex03_fuzzy_operations.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.3 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3()
