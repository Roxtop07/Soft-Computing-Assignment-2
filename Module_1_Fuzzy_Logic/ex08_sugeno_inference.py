"""
Practical 1.8: Sugeno (Takagi-Sugeno-Kang) Fuzzy Inference System
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To implement a simple Sugeno Fuzzy Inference System using Python.

Theory:
Introduced by Tomohiro Takagi and Michio Sugeno (1985), the Sugeno (or TSK) model uses
mathematical functions in the rule consequents rather than fuzzy sets.
- 0th-Order Sugeno: Consequent is a singleton constant:
      Rule i: IF x is A_i AND y is B_i THEN z_i = k_i
- 1st-Order Sugeno: Consequent is a linear function:
      Rule i: IF x is A_i AND y is B_i THEN z_i = p_i * x + q_i * y + r_i

Final Defuzzified Output (Weighted Average):
      z* = (∑ w_i * z_i) / (∑ w_i)
where w_i = min(μ_Ai(x), μ_Bi(y)) is the rule firing strength.
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

def sugeno_first_order(x, y):
    """
    Evaluates 1st-order Sugeno model with 4 rules:
    x in [0, 10]: Small (0,0,5), Large (5,10,10)
    y in [0, 10]: Small (0,0,5), Large (5,10,10)
    Consequents:
    Rule 1: IF x is Small AND y is Small THEN z1 = -x + y + 1
    Rule 2: IF x is Small AND y is Large THEN z2 = -y + 3
    Rule 3: IF x is Large AND y is Small THEN z3 = -x + 3
    Rule 4: IF x is Large AND y is Large THEN z4 = x + y + 2
    """
    u = np.linspace(0, 10, 100)
    mu_x_small = float(np.interp(x, u, trimf(u, 0, 0, 10)))
    mu_x_large = float(np.interp(x, u, trimf(u, 0, 10, 10)))
    mu_y_small = float(np.interp(y, u, trimf(u, 0, 0, 10)))
    mu_y_large = float(np.interp(y, u, trimf(u, 0, 10, 10)))

    # Weights
    w1 = min(mu_x_small, mu_y_small)
    w2 = min(mu_x_small, mu_y_large)
    w3 = min(mu_x_large, mu_y_small)
    w4 = min(mu_x_large, mu_y_large)

    # Consequent equations
    z1 = -x + y + 1
    z2 = -y + 3
    z3 = -x + 3
    z4 = x + y + 2

    weights = [w1, w2, w3, w4]
    consequents = [z1, z2, z3, z4]

    w_sum = sum(weights)
    if w_sum == 0:
        return 0.0, weights, consequents
    z_star = sum(w * z for w, z in zip(weights, consequents)) / w_sum
    return z_star, weights, consequents

def run_practical_8():
    print("=" * 60)
    print("PRACTICAL 1.8: Sugeno Fuzzy Inference System (TSK)")
    print("=" * 60)

    test_points = [(2.0, 3.0), (7.0, 2.0), (8.0, 8.0), (5.0, 5.0)]

    for x_val, y_val in test_points:
        z_out, w, z = sugeno_first_order(x_val, y_val)
        print(f"Inputs: (x={x_val:.1f}, y={y_val:.1f}) -> Weights: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}]")
        print(f"  -> Sugeno Weighted Average Output z* = {z_out:.3f}")

    # Generate 3D Response Surface
    grid_x = np.linspace(0, 10, 30)
    grid_y = np.linspace(0, 10, 30)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j], _, _ = sugeno_first_order(X[i, j], Y[i, j])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)
    ax.set_title("1st-Order Sugeno FIS Output Response Surface", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Input X", labelpad=10)
    ax.set_ylabel("Input Y", labelpad=10)
    ax.set_zlabel("Output Z*", labelpad=10)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Z*')
    ax.view_init(elev=28, azim=130)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex08_sugeno_inference.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.8 completed successfully.\n")

if __name__ == "__main__":
    run_practical_8()
