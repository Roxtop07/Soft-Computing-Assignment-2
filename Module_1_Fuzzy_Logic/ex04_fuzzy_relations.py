"""
Practical 1.4: Fuzzy Relations and Max-Min Composition
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To create and display a simple fuzzy relation between two fuzzy sets using Python.

Theory:
A fuzzy relation R between universe X and universe Y is a fuzzy subset of the Cartesian product X × Y:
    R = { ((x, y), μ_R(x, y)) | (x, y) ∈ X × Y }
where μ_R(x, y) = min(μ_A(x), μ_B(y))  (Mamdani product/min relation).

Given another relation S on Y × Z, the Max-Min Composition (T = R ∘ S) is defined as:
    μ_T(x, z) = max_{y ∈ Y} [ min(μ_R(x, y), μ_S(y, z)) ]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def max_min_composition(R, S):
    """Computes Max-Min composition T = R o S"""
    n_x, n_y = R.shape
    n_y2, n_z = S.shape
    assert n_y == n_y2, "Inner dimensions must match for relation composition"
    T = np.zeros((n_x, n_z))
    for i in range(n_x):
        for k in range(n_z):
            T[i, k] = np.max(np.minimum(R[i, :], S[:, k]))
    return T

def run_practical_4():
    print("=" * 60)
    print("PRACTICAL 1.4: Fuzzy Relations & Composition")
    print("=" * 60)

    # Universes: X = {x1, x2, x3}, Y = {y1, y2, y3, y4}
    x_labels = ['x1', 'x2', 'x3']
    y_labels = ['y1', 'y2', 'y3', 'y4']
    z_labels = ['z1', 'z2', 'z3']

    # Fuzzy set A on X and B on Y
    A = np.array([0.2, 0.7, 1.0])
    B = np.array([0.4, 0.9, 0.6, 0.1])

    # 1. Fuzzy Cartesian Relation R = A x B via min operator
    R = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            R[i, j] = min(A[i], B[j])

    print("\nFuzzy Set A:", A)
    print("Fuzzy Set B:", B)
    print("\nFuzzy Relation Matrix R (A × B, min operator):")
    print(R)

    # 2. Another relation S on Y x Z
    S = np.array([
        [0.6, 0.3, 0.8],
        [0.2, 0.9, 0.5],
        [0.8, 0.4, 0.1],
        [0.3, 0.7, 0.9]
    ])
    print("\nFuzzy Relation Matrix S (Y × Z):")
    print(S)

    # 3. Max-Min Composition T = R o S
    T = max_min_composition(R, S)
    print("\nMax-Min Composition T = R ∘ S (X × Z):")
    print(T)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Heatmap R
    im1 = axes[0].imshow(R, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title("Fuzzy Relation R (X × Y)", fontweight='bold')
    axes[0].set_xticks(range(len(y_labels)))
    axes[0].set_xticklabels(y_labels)
    axes[0].set_yticks(range(len(x_labels)))
    axes[0].set_yticklabels(x_labels)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            axes[0].text(j, i, f"{R[i, j]:.2f}", ha='center', va='center', color='black')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Heatmap S
    im2 = axes[1].imshow(S, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title("Fuzzy Relation S (Y × Z)", fontweight='bold')
    axes[1].set_xticks(range(len(z_labels)))
    axes[1].set_xticklabels(z_labels)
    axes[1].set_yticks(range(len(y_labels)))
    axes[1].set_yticklabels(y_labels)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            axes[1].text(j, i, f"{S[i, j]:.2f}", ha='center', va='center', color='black')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Heatmap T = R o S
    im3 = axes[2].imshow(T, cmap='Oranges', vmin=0, vmax=1)
    axes[2].set_title("Max-Min Composition T = R ∘ S", fontweight='bold')
    axes[2].set_xticks(range(len(z_labels)))
    axes[2].set_xticklabels(z_labels)
    axes[2].set_yticks(range(len(x_labels)))
    axes[2].set_yticklabels(x_labels)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            axes[2].text(j, i, f"{T[i, j]:.2f}", ha='center', va='center', color='black')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex04_fuzzy_relations.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.4 completed successfully.\n")

if __name__ == "__main__":
    run_practical_4()
