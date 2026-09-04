"""
Practical 1.9: Comparison of Defuzzification Methods
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To apply and compare basic defuzzification methods such as Centroid, Bisector, and Mean of Maximum using Python.

Theory:
Defuzzification converts a fuzzy set resulting from inference aggregation into a crisp scalar:
1. Centroid (Center of Gravity - COG):
       z* = (∫ z · μ(z) dz) / (∫ μ(z) dz)
2. Bisector of Area (BOA):
       The point z* that divides the area under μ(z) into two equal halves:
       ∫_{z_min}^{z*} μ(z) dz = ∫_{z*}^{z_max} μ(z) dz
3. Mean of Maximum (MOM):
       z* = (∑_{z ∈ M} z) / |M|  where M = {z | μ(z) = max_z μ(z)}
4. Smallest of Maximum (SOM): min(M)
5. Largest of Maximum (LOM): max(M)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import skfuzzy as fuzz
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a + 1e-12), (c - x)/(c - b + 1e-12)), 0.0)

def custom_defuzz(x, mfx, mode='centroid'):
    """Manual mathematical implementation of defuzzification methods."""
    if mode == 'centroid':
        return np.sum(x * mfx) / np.sum(mfx)
    elif mode == 'bisector':
        cum_area = np.cumsum(mfx)
        half_area = cum_area[-1] / 2.0
        idx = np.searchsorted(cum_area, half_area)
        return x[idx]
    elif mode == 'mom':
        max_val = np.max(mfx)
        max_indices = np.where(mfx >= max_val - 1e-6)[0]
        return np.mean(x[max_indices])
    elif mode == 'som':
        max_val = np.max(mfx)
        max_indices = np.where(mfx >= max_val - 1e-6)[0]
        return x[max_indices[0]]
    elif mode == 'lom':
        max_val = np.max(mfx)
        max_indices = np.where(mfx >= max_val - 1e-6)[0]
        return x[max_indices[-1]]
    else:
        raise ValueError(f"Unknown mode {mode}")

def run_practical_9():
    print("=" * 60)
    print("PRACTICAL 1.9: Defuzzification Methods Benchmark")
    print("=" * 60)

    # Output universe
    x = np.linspace(0, 10, 1000)

    # Construct an asymmetric, multi-peak aggregated fuzzy set
    mf1 = 0.6 * trimf(x, 0, 2, 5)
    mf2 = 0.9 * trimf(x, 3, 6, 8)
    mf3 = 0.4 * trimf(x, 7, 9, 10)
    mfx = np.maximum(mf1, np.maximum(mf2, mf3))

    methods = ['centroid', 'bisector', 'mom', 'som', 'lom']
    results = {}

    print("\nDefuzzification Calculation Results:")
    print("-" * 45)
    for m in methods:
        if HAS_SKFUZZY and m in ['centroid', 'bisector', 'mom', 'som', 'lom']:
            val = fuzz.defuzz(x, mfx, m)
        else:
            val = custom_defuzz(x, mfx, m)
        results[m] = val
        print(f"  {m.upper():<12} Defuzzification -> z* = {val:.4f}")

    # Visualization
    plt.figure(figsize=(10, 5.5))
    plt.plot(x, mfx, 'k-', linewidth=2, label='Aggregated Fuzzy Set μ(z)')
    plt.fill_between(x, mfx, color='lightgray', alpha=0.5)

    colors = {
        'centroid': '#e74c3c',
        'bisector': '#3498db',
        'mom': '#2ecc71',
        'som': '#9b59b6',
        'lom': '#f39c12'
    }

    for m, val in results.items():
        plt.axvline(x=val, color=colors[m], linestyle='--', linewidth=2,
                    label=f"{m.upper()} (z* = {val:.2f})")

    plt.title("Comparative Analysis of Defuzzification Methods", fontsize=12, fontweight='bold')
    plt.xlabel("Universe of Discourse Z", fontsize=11)
    plt.ylabel("Degree of Membership μ(z)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', framealpha=0.95)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex09_defuzzification_methods.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.9 completed successfully.\n")

if __name__ == "__main__":
    run_practical_9()
