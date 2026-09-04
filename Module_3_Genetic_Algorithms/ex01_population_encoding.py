"""
Practical 3.1: Population Representation and Encoding Methods
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To create and represent a population of individuals using different encoding methods in Python.

Theory:
In Evolutionary Computing, candidate solutions are encoded into chromosomes.
The primary representation schemes include:
1. Binary Encoding: Chromosomes are strings of bits ∈ {0, 1}. Standard for discrete problems.
2. Real-Valued (Continuous) Encoding: Chromosomes are vectors of floating-point numbers ∈ [a, b].
   Naturally represents continuous optimization variables without quantization error.
3. Permutation Encoding: Chromosomes are ordered permutations of integers [0, 1, ..., n-1].
   Mandatory for ordering/routing problems like the Traveling Salesperson Problem (TSP).
4. Gray Code Encoding: Reflected binary code where consecutive integers differ by exactly one bit,
   eliminating the "Hamming Cliff" phenomenon in standard binary encoding.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def binary_to_gray(n):
    """Converts standard binary integer to Gray code integer."""
    return n ^ (n >> 1)

def int_to_bitstring(val, bits=6):
    return [int(b) for b in format(val, f'0{bits}b')]

def generate_binary_population(pop_size, chrom_length):
    return np.random.randint(0, 2, size=(pop_size, chrom_length))

def generate_real_population(pop_size, dimensions, lower_bound=-5.0, upper_bound=5.0):
    return np.random.uniform(lower_bound, upper_bound, size=(pop_size, dimensions))

def generate_permutation_population(pop_size, num_cities):
    return np.array([np.random.permutation(num_cities) for _ in range(pop_size)])

def generate_gray_population(pop_size, bits=6):
    integers = np.random.randint(0, 2**bits, size=pop_size)
    gray_pop = [int_to_bitstring(binary_to_gray(val), bits) for val in integers]
    return np.array(gray_pop), integers

def run_practical_3_1():
    print("=" * 60)
    print("PRACTICAL 3.1: Genetic Algorithm Population Encodings")
    print("=" * 60)

    pop_size = 6
    np.random.seed(42)

    # 1. Binary Encoding
    bin_pop = generate_binary_population(pop_size, chrom_length=8)
    print("\n1. Binary Encoded Population (Length 8 bits):")
    for i, ind in enumerate(bin_pop):
        print(f"   Ind {i+1}: {ind.tolist()} -> Decimal Value = {int(''.join(map(str, ind)), 2)}")

    # 2. Real-Valued Encoding
    real_pop = generate_real_population(pop_size, dimensions=3, lower_bound=-10.0, upper_bound=10.0)
    print("\n2. Real-Valued Encoded Population (3D Continuous Variables):")
    for i, ind in enumerate(real_pop):
        print(f"   Ind {i+1}: [{ind[0]:6.3f}, {ind[1]:6.3f}, {ind[2]:6.3f}]")

    # 3. Permutation Encoding
    perm_pop = generate_permutation_population(pop_size, num_cities=5)
    print("\n3. Permutation Encoded Population (5-City Tour for TSP):")
    for i, ind in enumerate(perm_pop):
        print(f"   Ind {i+1}: Tour = {ind.tolist()}")

    # 4. Gray Code vs Binary (Hamming Cliff demonstration)
    print("\n4. Hamming Distance Comparison: Standard Binary vs. Gray Code (Transition 3 -> 4):")
    bin_3 = format(3, '04b')
    bin_4 = format(4, '04b')
    gray_3 = format(binary_to_gray(3), '04b')
    gray_4 = format(binary_to_gray(4), '04b')
    bin_hd = sum(c1 != c2 for c1, c2 in zip(bin_3, bin_4))
    gray_hd = sum(c1 != c2 for c1, c2 in zip(gray_3, gray_4))
    print(f"   Decimal 3 ({bin_3}) -> 4 ({bin_4}) in Binary: Hamming Distance = {bin_hd} (Hamming Cliff!)")
    print(f"   Decimal 3 ({gray_3}) -> 4 ({gray_4}) in Gray Code: Hamming Distance = {gray_hd} (Single-bit transition)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Binary Population Grid
    axes[0].imshow(bin_pop, cmap='binary', aspect='auto')
    axes[0].set_title("Binary Encoded Population (0/1)", fontweight='bold')
    axes[0].set_xlabel("Gene Locus")
    axes[0].set_ylabel("Individual ID")

    # Real-Valued Scatter
    axes[1].scatter(real_pop[:, 0], real_pop[:, 1], c='purple', s=120, edgecolors='black')
    for i in range(pop_size):
        axes[1].annotate(f"Ind {i+1}", (real_pop[i, 0]+0.3, real_pop[i, 1]+0.3))
    axes[1].set_title("Real-Valued 2D Solution Distribution", fontweight='bold')
    axes[1].set_xlabel("Variable x1")
    axes[1].set_ylabel("Variable x2")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Permutation Matrix Visualization
    axes[2].imshow(perm_pop, cmap='viridis', aspect='auto')
    for i in range(pop_size):
        for j in range(5):
            axes[2].text(j, i, str(perm_pop[i, j]), ha='center', va='center', color='white', fontweight='bold')
    axes[2].set_title("Permutation Order (TSP Routing)", fontweight='bold')
    axes[2].set_xlabel("Visit Step")
    axes[2].set_ylabel("Individual ID")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex01_population_encoding.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.1 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_1()
