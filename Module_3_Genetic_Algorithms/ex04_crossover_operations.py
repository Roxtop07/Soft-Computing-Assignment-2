"""
Practical 3.4: Crossover (Recombination) Operators
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To implement the crossover operation for generating new offspring from selected individuals using Python.

Theory:
Crossover combines the genetic material of two parent chromosomes to produce offspring:
1. Single-Point Crossover: A crossover point k is chosen at random; genes beyond k are exchanged.
2. Two-Point Crossover: Two points k1 < k2 are chosen; genes between k1 and k2 are exchanged,
   reducing positional bias.
3. Uniform Crossover: Each gene is inherited from parent 1 or parent 2 based on a random binary mask.
4. Order Crossover (OX1): Specialized for permutation representations (e.g., TSP).
   Preserves relative order and prevents duplicate allele entries.
5. Arithmetic Crossover: For continuous/real-valued encodings:
       c1 = α * p1 + (1 - α) * p2
       c2 = (1 - α) * p1 + α * p2
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def single_point_crossover(p1, p2, point=None):
    n = len(p1)
    k = np.random.randint(1, n) if point is None else point
    c1 = np.concatenate([p1[:k], p2[k:]])
    c2 = np.concatenate([p2[:k], p1[k:]])
    return c1, c2, k

def two_point_crossover(p1, p2, points=None):
    n = len(p1)
    if points is None:
        pts = np.sort(np.random.choice(range(1, n), size=2, replace=False))
        k1, k2 = pts[0], pts[1]
    else:
        k1, k2 = points
    c1 = np.concatenate([p1[:k1], p2[k1:k2], p1[k2:]])
    c2 = np.concatenate([p2[:k1], p1[k1:k2], p2[k2:]])
    return c1, c2, (k1, k2)

def uniform_crossover(p1, p2, p_swap=0.5, mask=None):
    n = len(p1)
    if mask is None:
        mask = np.random.rand(n) < p_swap
    c1 = np.where(mask, p2, p1)
    c2 = np.where(mask, p1, p2)
    return c1, c2, mask

def order_crossover_ox1(p1, p2, points=None):
    n = len(p1)
    if points is None:
        pts = np.sort(np.random.choice(range(1, n), size=2, replace=False))
        k1, k2 = pts[0], pts[1]
    else:
        k1, k2 = points

    c1 = [None] * n
    c1[k1:k2] = p1[k1:k2]
    p2_remaining = [item for item in p2 if item not in c1[k1:k2]]
    ptr = 0
    for i in range(n):
        if c1[i] is None:
            c1[i] = p2_remaining[ptr]
            ptr += 1
    return np.array(c1), (k1, k2)

def run_practical_3_4():
    print("=" * 60)
    print("PRACTICAL 3.4: Crossover (Recombination) Operations")
    print("=" * 60)

    # Binary Parents
    p1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    p2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    print("Parent 1:", p1.tolist())
    print("Parent 2:", p2.tolist())

    # 1. Single-Point
    c1_sp, c2_sp, pt_sp = single_point_crossover(p1, p2, point=4)
    print(f"\n1. Single-Point Crossover (Cut at index {pt_sp}):")
    print(f"   Offspring 1: {c1_sp.tolist()}")
    print(f"   Offspring 2: {c2_sp.tolist()}")

    # 2. Two-Point
    c1_tp, c2_tp, pts_tp = two_point_crossover(p1, p2, points=(2, 6))
    print(f"\n2. Two-Point Crossover (Cuts at {pts_tp}):")
    print(f"   Offspring 1: {c1_tp.tolist()}")
    print(f"   Offspring 2: {c2_tp.tolist()}")

    # 3. Uniform Crossover
    mask = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    c1_un, c2_un, _ = uniform_crossover(p1, p2, mask=mask)
    print(f"\n3. Uniform Crossover (Mask: {mask.astype(int).tolist()}):")
    print(f"   Offspring 1: {c1_un.tolist()}")
    print(f"   Offspring 2: {c2_un.tolist()}")

    # 4. Order Crossover (OX1) for TSP Permutation
    perm_p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    perm_p2 = np.array([8, 7, 6, 5, 4, 3, 2, 1])
    c1_ox, cuts_ox = order_crossover_ox1(perm_p1, perm_p2, points=(3, 6))
    print(f"\n4. Order Crossover OX1 (Permutations, cuts {cuts_ox}):")
    print(f"   Permutation Parent 1: {perm_p1.tolist()}")
    print(f"   Permutation Parent 2: {perm_p2.tolist()}")
    print(f"   Valid Offspring (OX1): {c1_ox.tolist()} (No duplicates, valid tour!)")

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    def plot_chrom(ax, chrom, title, highlight_range=None, highlight_mask=None):
        data = np.array([chrom])
        im = ax.imshow(data, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks([])
        ax.set_xticks(range(len(chrom)))
        ax.set_xticklabels([f"g{i}" for i in range(len(chrom))])
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        for i, val in enumerate(chrom):
            ax.text(i, 0, str(val), ha='center', va='center', color='white' if val==1 else 'black',
                    fontweight='bold', fontsize=12)
        if highlight_range:
            k1, k2 = highlight_range
            ax.axvline(k1 - 0.5, color='yellow', linestyle='--', linewidth=2.5)
            ax.axvline(k2 - 0.5, color='yellow', linestyle='--', linewidth=2.5)

    plot_chrom(axes[0], p1, "Parent 1 [All 1s]")
    plot_chrom(axes[1], p2, "Parent 2 [All 0s]")
    plot_chrom(axes[2], c1_sp, f"Single-Point Offspring 1 (Cut at locus 4)", highlight_range=(4, 4))
    plot_chrom(axes[3], c1_tp, f"Two-Point Offspring 1 (Cuts at loci 2, 6)", highlight_range=(2, 6))

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex04_crossover_operators.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.4 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_4()
