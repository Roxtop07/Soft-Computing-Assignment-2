"""
Practical 1.1: Simple Fuzzy Sets in Python
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To create and visualize simple fuzzy sets in Python using membership values.

Theory:
A classical (crisp) set A in a universe of discourse X is defined by a characteristic function:
    χ_A(x) = 1 if x ∈ A, and 0 if x ∉ A.
In contrast, a Fuzzy Set Ã is characterized by a membership function:
    μ_Ã(x) : X -> [0, 1]
where each element x is associated with a degree of membership between 0 and 1,
representing the degree of truth or belongingness.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_practical_1():
    print("=" * 60)
    print("PRACTICAL 1.1: Simple Fuzzy Sets Creation & Visualization")
    print("=" * 60)

    # Discrete universe: Age groups from 0 to 80 years
    ages = np.array([10, 20, 25, 30, 35, 40, 50, 60, 70, 80])

    # Crisp set: "Young" (age <= 30)
    crisp_young = np.array([1 if a <= 30 else 0 for a in ages])

    # Fuzzy set: "Young" with continuous grades of membership
    # Using sigmoidal / decay membership curve: μ(x) = 1 / (1 + ((x - 20)/10)^2)
    fuzzy_young = np.array([1.0, 1.0, 0.9, 0.7, 0.45, 0.2, 0.05, 0.0, 0.0, 0.0])

    # Fuzzy set: "Middle-Aged"
    fuzzy_middle = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 1.0, 0.8, 0.4, 0.1, 0.0])

    # Fuzzy set: "Old"
    fuzzy_old = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.8, 0.95, 1.0])

    print("\nDiscrete Universe (Age):", ages)
    print("Crisp Set 'Young':", crisp_young)
    print("Fuzzy Set 'Young' μ(x):", fuzzy_young)
    print("Fuzzy Set 'Middle' μ(x):", fuzzy_middle)
    print("Fuzzy Set 'Old' μ(x):", fuzzy_old)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Crisp vs Fuzzy comparison for "Young"
    ax1.step(ages, crisp_young, where='mid', color='crimson', label='Crisp Set "Young"', linewidth=2.5)
    ax1.plot(ages, fuzzy_young, 'bo-', label='Fuzzy Set "Young"', linewidth=2, markersize=7)
    ax1.set_title("Crisp vs. Fuzzy Representation: 'Young'", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Age (Years)", fontsize=11)
    ax1.set_ylabel("Degree of Membership μ(x)", fontsize=11)
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot 2: Multiple Fuzzy Sets over Universe
    ax2.plot(ages, fuzzy_young, 'bo-', label='Young', linewidth=2)
    ax2.plot(ages, fuzzy_middle, 'g^-', label='Middle-Aged', linewidth=2)
    ax2.plot(ages, fuzzy_old, 'rs-', label='Old', linewidth=2)
    ax2.set_title("Linguistic Partitions: Age Classification", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Age (Years)", fontsize=11)
    ax2.set_ylabel("Degree of Membership μ(x)", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex01_simple_fuzzy_sets.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.1 completed successfully.\n")

if __name__ == "__main__":
    run_practical_1()
