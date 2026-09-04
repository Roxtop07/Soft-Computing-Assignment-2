"""
Practical 3.9: Differential Evolution (DE) Optimization Algorithm
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To implement a simple Differential Evolution (DE) algorithm for solving an optimization problem using Python.

Theory:
Differential Evolution (Rainer Storn and Kenneth Price, 1997) is an exceptionally robust global optimizer
operating on continuous real-valued spaces.
The canonical strategy is DE/rand/1/bin:
1. Mutation: For each target vector x_i, select 3 mutually distinct random agents r1 ≠ r2 ≠ r3 ≠ i:
       v_i = x_r1 + F * (x_r2 - x_r3)
   where F ∈ [0, 2] is the differential weight / scaling factor.
2. Binomial Crossover: Yields trial vector u_i:
       u_{i, j} = v_{i, j}  if (rand(0,1) <= CR  or  j == j_rand)  else  x_{i, j}
   where CR ∈ [0, 1] is the crossover probability, and j_rand ensures at least one mutated allele is transferred.
3. One-to-One Greedy Selection:
       x_i(t+1) = u_i  if f(u_i) <= f(x_i)  else  x_i
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Complex Multimodal Benchmark: Ackley Function
# Global minimum: f(0, 0, ...) = 0
def ackley(X):
    d = X.shape[-1]
    sum_sq = np.sum(X**2, axis=-1)
    sum_cos = np.sum(np.cos(2 * np.pi * X), axis=-1)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + 20.0 + np.e

class DifferentialEvolution:
    def __init__(self, pop_size=40, dim=2, bounds=(-5.0, 5.0),
                 F=0.8, CR=0.9, max_gen=80):
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.F = F
        self.CR = CR
        self.max_gen = max_gen

    def optimize(self):
        np.random.seed(42)
        lb, ub = self.bounds

        # Initialize population
        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = ackley(pop)

        best_idx = np.argmin(fitness)
        best_agent = pop[best_idx].copy()
        best_cost = fitness[best_idx]
        history = [best_cost]

        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # Select 3 candidates different from i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)

                # Binomial Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fit = ackley(trial)
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit

                    if trial_fit < best_cost:
                        best_cost = trial_fit
                        best_agent = trial.copy()

            history.append(best_cost)

        return best_agent, best_cost, history

def run_practical_3_9():
    print("=" * 60)
    print("PRACTICAL 3.9: Differential Evolution (DE/rand/1/bin)")
    print("=" * 60)

    de = DifferentialEvolution(pop_size=40, dim=2, bounds=(-5.0, 5.0), F=0.8, CR=0.85, max_gen=75)
    best_x, best_fit, history = de.optimize()

    print(f"Target Problem: 2D Ackley Optimization (Highly Multimodal Surface)")
    print(f"Global Optimum: x* = (0, 0) with f(x*) = 0.0")
    print(f"\nDE Discovered Solution: x* = ({best_x[0]:.6f}, {best_x[1]:.6f})")
    print(f"Optimal Fitness: f(x*)   = {best_fit:.8f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Convergence Plot
    ax1.plot(range(len(history)), history, 'b-', linewidth=2.2)
    ax1.set_yscale('log')
    ax1.set_title("Differential Evolution Convergence on Ackley Function", fontweight='bold')
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Best Cost (Log Scale)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 3D Ackley Landscape
    x_span = np.linspace(-5.0, 5.0, 100)
    y_span = np.linspace(-5.0, 5.0, 100)
    XX, YY = np.meshgrid(x_span, y_span)
    grid = np.stack([XX, YY], axis=-1)
    ZZ = ackley(grid)

    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(XX, YY, ZZ, cmap='inferno', edgecolor='none', alpha=0.85)
    ax2.scatter(best_x[0], best_x[1], best_fit, color='cyan', s=150, marker='*', label='DE Optimum')
    ax2.set_title("Ackley Multi-Peak Benchmark Landscape", fontweight='bold')
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Cost")
    ax2.view_init(elev=35, azim=215)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex09_differential_evolution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.9 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_9()
