"""
Practical 3.5: Mutation Operators in Genetic Algorithms
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To implement the mutation operation for introducing variations in individuals using Python.

Theory:
Mutation is an exploration operator that maintains genetic diversity in the population and prevents
premature convergence to suboptimal local extrema.
Common Mutation Schemes:
1. Bit-Flip Mutation (Binary): Each gene locus is inverted with a small mutation probability P_m (typically 1/L).
2. Swap Mutation (Permutations): Selects two gene loci at random and swaps their positions (preserves permutation validity).
3. Inversion Mutation (Permutations): Reverses the order of genes between two randomly chosen indices.
4. Gaussian Mutation (Continuous / Real-Valued): Adds zero-mean Gaussian perturbation:
       x_mutated = x + N(0, σ^2)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def bit_flip_mutation(chromosome, pm=0.1):
    mutated = chromosome.copy()
    flips = np.random.rand(len(chromosome)) < pm
    mutated[flips] = 1 - mutated[flips]
    return mutated, flips

def swap_mutation(permutation):
    mutated = permutation.copy()
    idx1, idx2 = np.random.choice(len(permutation), size=2, replace=False)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated, (idx1, idx2)

def inversion_mutation(permutation):
    mutated = permutation.copy()
    pts = np.sort(np.random.choice(len(permutation), size=2, replace=False))
    k1, k2 = pts[0], pts[1]
    mutated[k1:k2+1] = mutated[k1:k2+1][::-1]
    return mutated, (k1, k2)

def gaussian_mutation(real_vector, pm=0.3, sigma=0.5, bounds=(-5.0, 5.0)):
    mutated = real_vector.copy()
    mask = np.random.rand(len(real_vector)) < pm
    noise = np.random.normal(0, sigma, size=len(real_vector))
    mutated[mask] += noise[mask]
    mutated = np.clip(mutated, bounds[0], bounds[1])
    return mutated, mask

def run_practical_3_5():
    print("=" * 60)
    print("PRACTICAL 3.5: Mutation Operators Implementation")
    print("=" * 60)

    np.random.seed(42)

    # 1. Bit-Flip Mutation
    orig_bin = np.zeros(10, dtype=int)
    mut_bin, flips = bit_flip_mutation(orig_bin, pm=0.3)
    print("1. Bit-Flip Mutation (Binary chromosome):")
    print(f"   Original: {orig_bin.tolist()}")
    print(f"   Mutated:  {mut_bin.tolist()} (Flipped positions: {np.where(flips)[0].tolist()})")

    # 2. Swap Mutation
    orig_perm = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    mut_swap, (s1, s2) = swap_mutation(orig_perm)
    print("\n2. Swap Mutation (Permutation for TSP):")
    print(f"   Original: {orig_perm.tolist()}")
    print(f"   Mutated:  {mut_swap.tolist()} (Swapped index {s1} [val={orig_perm[s1]}] with index {s2} [val={orig_perm[s2]}])")

    # 3. Inversion Mutation
    mut_inv, (i1, i2) = inversion_mutation(orig_perm)
    print("\n3. Inversion Mutation (Permutation):")
    print(f"   Original: {orig_perm.tolist()}")
    print(f"   Mutated:  {mut_inv.tolist()} (Reversed segment [{i1}:{i2}])")

    # 4. Gaussian Mutation
    orig_real = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    mut_real, r_mask = gaussian_mutation(orig_real, pm=0.6, sigma=1.2)
    print("\n4. Gaussian Mutation (Real-Valued Vector):")
    print(f"   Original: {orig_real.tolist()}")
    print(f"   Mutated:  {[round(x, 4) for x in mut_real.tolist()]}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    # Binary mutation visualization
    axes[0, 0].stem(range(len(orig_bin)), mut_bin, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0, 0].set_title("Bit-Flip Mutation (1 indicates mutated locus)", fontweight='bold')
    axes[0, 0].set_xlabel("Gene Locus")
    axes[0, 0].set_ylabel("Allele Value")
    axes[0, 0].set_ylim(-0.1, 1.2)

    # Permutation Swap visualization
    axes[0, 1].plot(range(len(orig_perm)), orig_perm, 'ko--', label='Original Tour')
    axes[0, 1].plot(range(len(mut_swap)), mut_swap, 'ro-', label='Swap Mutated Tour')
    axes[0, 1].scatter([s1, s2], [mut_swap[s1], mut_swap[s2]], color='yellow', s=150, zorder=5, edgecolors='k')
    axes[0, 1].set_title("Swap Mutation: Tour Alteration", fontweight='bold')
    axes[0, 1].set_xlabel("Order Position")
    axes[0, 1].set_ylabel("City ID")
    axes[0, 1].legend()

    # Inversion Mutation visualization
    axes[1, 0].plot(range(len(orig_perm)), orig_perm, 'ko--', label='Original')
    axes[1, 0].plot(range(len(mut_inv)), mut_inv, 'go-', label='Inverted')
    axes[1, 0].axvspan(i1, i2, color='green', alpha=0.2, label=f'Inverted Segment [{i1}:{i2}]')
    axes[1, 0].set_title("Inversion Mutation: Segment Reversal", fontweight='bold')
    axes[1, 0].set_xlabel("Order Position")
    axes[1, 0].set_ylabel("City ID")
    axes[1, 0].legend()

    # Gaussian Mutation perturbation distribution
    samples = [gaussian_mutation(np.array([0.0]), pm=1.0, sigma=1.0)[0][0] for _ in range(2000)]
    axes[1, 1].hist(samples, bins=35, density=True, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("Gaussian Mutation Perturbation Distribution N(0, 1)", fontweight='bold')
    axes[1, 1].set_xlabel("Δx Perturbation")
    axes[1, 1].set_ylabel("Probability Density")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex05_mutation_operators.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.5 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_5()
