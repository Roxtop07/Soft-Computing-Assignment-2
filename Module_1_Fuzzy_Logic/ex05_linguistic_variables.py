"""
Practical 1.5: Fuzzy Linguistic Variables (Low, Medium, High)
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To create fuzzy linguistic variables such as Low, Medium, and High using membership functions in Python.

Theory:
According to Lotfi Zadeh, a Linguistic Variable is a variable whose values are words or sentences in a
natural or artificial language, rather than numbers.
Formally defined as a quintuple (x, T(x), U, G, M):
- x: Name of the variable (e.g., 'Vehicle Speed')
- T(x): Linguistic term set {Low, Medium, High}
- U: Universe of discourse [0, 120] km/h
- G: Syntactic grammar
- M: Semantic rule mapping each linguistic term to a fuzzy set over U.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def trapmf(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a)/(b - a + 1e-12), 1.0), (d - x)/(d - c + 1e-12)), 0.0)

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a + 1e-12), (c - x)/(c - b + 1e-12)), 0.0)

def run_practical_5():
    print("=" * 60)
    print("PRACTICAL 1.5: Fuzzy Linguistic Variables Definition")
    print("=" * 60)

    # Linguistic Variable: Vehicle Speed (km/h)
    # Universe U = [0, 120]
    speed_universe = np.linspace(0, 120, 600)

    # Term Set: {Very Low, Low, Medium, High, Very High}
    terms = {
        'Very Low': trapmf(speed_universe, 0, 0, 15, 30),
        'Low': trimf(speed_universe, 20, 40, 60),
        'Medium': trimf(speed_universe, 45, 65, 85),
        'High': trimf(speed_universe, 70, 90, 105),
        'Very High': trapmf(speed_universe, 95, 110, 120, 120)
    }

    # Query speed crisp value
    test_speeds = [35.0, 60.0, 95.0]
    print("\nFuzzification of Crisp Speed Values:")
    print("-" * 55)
    for s in test_speeds:
        print(f"\nSpeed = {s} km/h:")
        for name, mf in terms.items():
            degree = float(np.interp(s, speed_universe, mf))
            if degree > 0.0:
                print(f"  -> {name:<10}: μ = {degree:.3f}")

    # Visualization
    plt.figure(figsize=(11, 5))
    palette = ['#2980b9', '#27ae60', '#f39c12', '#e67e22', '#c0392b']

    for (name, mf), color in zip(terms.items(), palette):
        plt.plot(speed_universe, mf, label=name, color=color, linewidth=2.5)
        plt.fill_between(speed_universe, mf, color=color, alpha=0.1)

    # Mark sample test point
    plt.axvline(x=60, color='black', linestyle='--', linewidth=1.5, label='Sample Probe (60 km/h)')
    plt.title("Linguistic Variable: Vehicle Speed (U = [0, 120] km/h)", fontsize=12, fontweight='bold')
    plt.xlabel("Speed (km/h)", fontsize=11)
    plt.ylabel("Degree of Membership μ(x)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center right', framealpha=0.95)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex05_linguistic_variables.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.5 completed successfully.\n")

if __name__ == "__main__":
    run_practical_5()
