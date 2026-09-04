"""
Practical 1.10: Design of a Complete Fuzzy Logic Controller (FLC)
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To design a simple Fuzzy Logic Controller using Python and scikit-fuzzy by combining
membership functions, fuzzy rules, fuzzy inference, and defuzzification.

Theory:
A standard 2-input 1-output industrial Fuzzy Logic Controller (FLC) regulates:
1. Input 1: Error e(t) = Setpoint - Process Variable
2. Input 2: Change in Error Δe(t) = e(t) - e(t - 1)
3. Output: Control Effort u(t)

Linguistic terms:
Negative Big (NB), Negative Small (NS), Zero (ZE), Positive Small (PS), Positive Big (PB).
Rule evaluation follows the standard MacVicar-Whelan rule matrix (25 rules).
Inference uses Mamdani Min implication and Centroid defuzzification.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a + 1e-12), (c - x)/(c - b + 1e-12)), 0.0)

# Universes: e in [-10, 10], de in [-5, 5], u in [-100, 100]
u_e = np.linspace(-10, 10, 300)
u_de = np.linspace(-5, 5, 300)
u_ctrl = np.linspace(-100, 100, 500)

# Partition definition dictionary
def get_partitions(u, span):
    w = span / 2.0
    return {
        'NB': trimf(u, -span, -span, -w),
        'NS': trimf(u, -span, -w, 0),
        'ZE': trimf(u, -w, 0, w),
        'PS': trimf(u, 0, w, span),
        'PB': trimf(u, w, span, span)
    }

mf_e = get_partitions(u_e, 10.0)
mf_de = get_partitions(u_de, 5.0)
mf_u = get_partitions(u_ctrl, 100.0)

# 25-Rule MacVicar-Whelan Rule Matrix
# Rows: Error (NB, NS, ZE, PS, PB)
# Cols: Change in Error (NB, NS, ZE, PS, PB)
RULE_MATRIX = {
    ('NB', 'NB'): 'NB', ('NB', 'NS'): 'NB', ('NB', 'ZE'): 'NB', ('NB', 'PS'): 'NS', ('NB', 'PB'): 'ZE',
    ('NS', 'NB'): 'NB', ('NS', 'NS'): 'NS', ('NS', 'ZE'): 'NS', ('NS', 'PS'): 'ZE', ('NS', 'PB'): 'PS',
    ('ZE', 'NB'): 'NB', ('ZE', 'NS'): 'NS', ('ZE', 'ZE'): 'ZE', ('ZE', 'PS'): 'PS', ('ZE', 'PB'): 'PB',
    ('PS', 'NB'): 'NS', ('PS', 'NS'): 'ZE', ('PS', 'ZE'): 'PS', ('PS', 'PS'): 'PS', ('PS', 'PB'): 'PB',
    ('PB', 'NB'): 'ZE', ('PB', 'NS'): 'PS', ('PB', 'ZE'): 'PB', ('PB', 'PS'): 'PB', ('PB', 'PB'): 'PB',
}

def flc_step(e_val, de_val):
    """Evaluates the FLC for crisp (e, de) and returns crisp control action u*."""
    # Fuzzify
    mu_e = {k: float(np.interp(e_val, u_e, v)) for k, v in mf_e.items()}
    mu_de = {k: float(np.interp(de_val, u_de, v)) for k, v in mf_de.items()}

    aggregated = np.zeros_like(u_ctrl)

    for (term_e, term_de), term_u in RULE_MATRIX.items():
        w = min(mu_e[term_e], mu_de[term_de])
        if w > 0:
            clipped = np.minimum(w, mf_u[term_u])
            aggregated = np.maximum(aggregated, clipped)

    denom = np.sum(aggregated)
    if denom == 0:
        return 0.0
    return float(np.sum(u_ctrl * aggregated) / denom)

def run_practical_10():
    print("=" * 60)
    print("PRACTICAL 1.10: Complete Fuzzy Logic Controller (FLC)")
    print("=" * 60)

    # 1. Test specific points
    test_cases = [(4.5, -1.2), (-8.0, 0.5), (0.1, 0.0), (8.5, 3.2)]
    print("\nEvaluating FLC Output for Test States (Error, Change_in_Error):")
    print("-" * 55)
    for e_in, de_in in test_cases:
        u_out = flc_step(e_in, de_in)
        print(f"Error = {e_in:5.1f} | ΔError = {de_in:5.1f} -> Control Action u* = {u_out:6.2f}%")

    # 2. Compute 3D Control Surface
    e_grid = np.linspace(-10, 10, 25)
    de_grid = np.linspace(-5, 5, 25)
    E, DE = np.meshgrid(e_grid, de_grid)
    U = np.zeros_like(E)

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            U[i, j] = flc_step(E[i, j], DE[i, j])

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(E, DE, U, cmap='coolwarm', edgecolor='none', alpha=0.9)
    ax.set_title("FLC 3D Control Surface: Control Output u* vs (Error, ΔError)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Error e", labelpad=10)
    ax.set_ylabel("Change in Error Δe", labelpad=10)
    ax.set_zlabel("Control Output u (%)", labelpad=10)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Control Effort (%)')
    ax.view_init(elev=25, azim=215)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex10_flc_controller.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Control Surface Plot saved to: {plot_path}")
    print("[✓] Practical 1.10 completed successfully.\n")

if __name__ == "__main__":
    run_practical_10()
