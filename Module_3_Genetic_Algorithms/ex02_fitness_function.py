"""
Practical 3.2: Fitness Function Formulation and Evaluation
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To define and calculate a simple fitness function for evaluating individuals in a Genetic Algorithm using Python.

Theory:
The Fitness Function transforms an individual's chromosome into a non-negative scalar score
representing reproductive capability.
1. Minimization-to-Maximization Mapping:
   When minimizing an objective function g(x) >= 0:
       Fitness f(x) = 1.0 / (1.0 + g(x))   or   f(x) = C_max - g(x)
2. Benchmark Optimization Landscapes:
   - Sphere Function (Unimodal): f_sphere(x) = ∑ x_i^2, Global minimum at x* = 0
   - Rastrigin Function (Highly Multimodal with complex local traps):
       f_rastrigin(x) = 10*d + ∑ [x_i^2 - 10*cos(2π*x_i)]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sphere_objective(X):
    """X: shape (N, d)"""
    return np.sum(X ** 2, axis=-1)

def rastrigin_objective(X):
    """X: shape (N, d)"""
    d = X.shape[-1]
    return 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)

def fitness_from_cost(cost):
    """Maps cost (to minimize) into positive fitness (to maximize)."""
    return 1.0 / (1.0 + cost)

def run_practical_3_2():
    print("=" * 60)
    print("PRACTICAL 3.2: Fitness Function Definition & Evaluation")
    print("=" * 60)

    # Test candidate individuals on 2D Rastrigin function
    candidates = np.array([
        [0.0, 0.0],       # Global Optimum
        [1.0, 1.0],       # Near local minimum
        [2.5, -3.2],      # Suboptimal point
        [-4.5, 4.2]       # Poor individual near domain boundary
    ])

    costs = rastrigin_objective(candidates)
    fitnesses = fitness_from_cost(costs)

    print(f"{'Individual':^12} | {'Coordinates (x1, x2)':^24} | {'Cost g(x)':^12} | {'Fitness f(x)':^14}")
    print("-" * 70)
    for i in range(len(candidates)):
        coords = f"({candidates[i, 0]:5.2f}, {candidates[i, 1]:5.2f})"
        print(f"Ind {i+1:2d}       | {coords:^24} | {costs[i]:12.4f} | {fitnesses[i]:14.6f}")

    # Visualization: 3D Rastrigin Fitness Landscape
    x1 = np.linspace(-5.12, 5.12, 100)
    x2 = np.linspace(-5.12, 5.12, 100)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.stack([X1, X2], axis=-1)
    Z_cost = rastrigin_objective(grid)
    Z_fit = fitness_from_cost(Z_cost)

    fig = plt.figure(figsize=(13, 6))

    # Cost Landscape (3D Surface)
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, Z_cost, cmap='viridis', edgecolor='none', alpha=0.85)
    ax1.scatter(candidates[:, 0], candidates[:, 1], costs, color='red', s=60, label='Candidate Individuals')
    ax1.set_title("Objective Function: Rastrigin Surface (Minimize)", fontweight='bold')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("Cost")
    ax1.view_init(elev=35, azim=225)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Fitness Landscape (Inverted / Scaled to Maximize)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X1, X2, Z_fit, cmap='plasma', edgecolor='none', alpha=0.85)
    ax2.scatter(candidates[:, 0], candidates[:, 1], fitnesses, color='lime', s=60, label='Candidates')
    ax2.set_title("Fitness Landscape: f(x) = 1 / (1 + Cost)", fontweight='bold')
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Fitness")
    ax2.view_init(elev=35, azim=225)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex02_fitness_functions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.2 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_2()
