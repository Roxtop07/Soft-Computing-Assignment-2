"""
Practical 1.2: Triangular, Trapezoidal, and Gaussian Membership Functions
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To create and plot Triangular, Trapezoidal, and Gaussian membership functions using Python and scikit-fuzzy.

Theory:
1. Triangular MF: Defined by lower limit a, peak b, and upper limit c:
   μ(x; a, b, c) = max(min((x - a)/(b - a), (c - x)/(c - b)), 0)
2. Trapezoidal MF: Defined by lower limit a, lower plateau b, upper plateau c, upper limit d:
   μ(x; a, b, c, d) = max(min((x - a)/(b - a), 1, (d - x)/(d - c)), 0)
3. Gaussian MF: Defined by central mean c and standard deviation σ:
   μ(x; c, σ) = exp(-0.5 * ((x - c) / σ)^2)
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

# Custom robust implementations in case skfuzzy is unavailable
def custom_trimf(x, abc):
    a, b, c = abc
    return np.maximum(np.minimum((x - a)/(b - a + 1e-12), (c - x)/(c - b + 1e-12)), 0)

def custom_trapmf(x, abcd):
    a, b, c, d = abcd
    return np.maximum(np.minimum(np.minimum((x - a)/(b - a + 1e-12), 1.0), (d - x)/(d - c + 1e-12)), 0)

def custom_gaussmf(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def run_practical_2():
    print("=" * 60)
    print("PRACTICAL 1.2: Triangular, Trapezoidal, and Gaussian MFs")
    print("=" * 60)
    print(f"Using scikit-fuzzy backend: {HAS_SKFUZZY}")

    x = np.linspace(0, 100, 1000)

    # 1. Triangular MFs
    tri_params = [20, 50, 80]
    if HAS_SKFUZZY:
        y_tri = fuzz.trimf(x, tri_params)
    else:
        y_tri = custom_trimf(x, tri_params)

    # 2. Trapezoidal MFs
    trap_params = [15, 35, 65, 85]
    if HAS_SKFUZZY:
        y_trap = fuzz.trapmf(x, trap_params)
    else:
        y_trap = custom_trapmf(x, trap_params)

    # 3. Gaussian MFs
    mean, sigma = 50, 15
    if HAS_SKFUZZY:
        y_gauss = fuzz.gaussmf(x, mean, sigma)
    else:
        y_gauss = custom_gaussmf(x, mean, sigma)

    print("\nEvaluated MFs at key points (x = 35, 50, 65):")
    for pt in [35, 50, 65]:
        val_tri = float(np.interp(pt, x, y_tri))
        val_trap = float(np.interp(pt, x, y_trap))
        val_gauss = float(np.interp(pt, x, y_gauss))
        print(f"  x={pt:2d} -> Triangular: {val_tri:.3f}, Trapezoidal: {val_trap:.3f}, Gaussian: {val_gauss:.3f}")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Subplot 1: Triangular
    axes[0].plot(x, y_tri, 'b-', linewidth=2.5, label=f'Triangular {tri_params}')
    axes[0].set_title("Triangular Membership Function: trimf(x, [20, 50, 80])", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Membership μ(x)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')

    # Subplot 2: Trapezoidal
    axes[1].plot(x, y_trap, 'g-', linewidth=2.5, label=f'Trapezoidal {trap_params}')
    axes[1].set_title("Trapezoidal Membership Function: trapmf(x, [15, 35, 65, 85])", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Membership μ(x)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')

    # Subplot 3: Gaussian
    axes[2].plot(x, y_gauss, 'r-', linewidth=2.5, label=f'Gaussian (μ={mean}, σ={sigma})')
    axes[2].set_title(f"Gaussian Membership Function: gaussmf(x, mean={mean}, sigma={sigma})", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Universe of Discourse X", fontsize=11)
    axes[2].set_ylabel("Membership μ(x)")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex02_membership_functions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.2 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2()
