"""
Practical 3.8: Particle Swarm Optimization (PSO) Algorithm
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To implement a simple Particle Swarm Optimization (PSO) algorithm for solving an optimization problem using Python.

Theory:
Particle Swarm Optimization (James Kennedy and Russell Eberhart, 1995) is a nature-inspired metaheuristic
based on the social foraging behaviors of bird flocks and fish schools.
State equations for particle i at iteration t:
1. Velocity Update:
       v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
   where:
   - w: Inertia weight (balances exploration vs exploitation).
   - c1: Cognitive coefficient (pull toward individual historical best).
   - c2: Social coefficient (pull toward swarm-wide global best).
   - r1, r2: Uniform random vectors ∈ [0, 1].
2. Position Update:
       x_i(t+1) = x_i(t) + v_i(t+1)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target: 2D Rosenbrock Function (Banana valley): f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
# Global minimum: f(1, 1) = 0
def rosenbrock(X):
    x, y = X[..., 0], X[..., 1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

class ParticleSwarmOptimizer:
    def __init__(self, n_particles=30, dim=2, bounds=(-2.0, 2.0),
                 w=0.7, c1=1.5, c2=1.5, n_iter=60):
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_iter = n_iter

    def optimize(self):
        np.random.seed(42)
        lb, ub = self.bounds

        # Initialize positions & velocities
        X = np.random.uniform(lb, ub, size=(self.n_particles, self.dim))
        V = np.random.uniform(-0.5, 0.5, size=(self.n_particles, self.dim))

        pbest = X.copy()
        pbest_val = rosenbrock(pbest)

        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        history_gbest = [gbest_val]
        particle_trajectories = [X.copy()]

        for it in range(self.n_iter):
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            # Update velocity & position
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            # Velocity clamping
            V = np.clip(V, -1.0, 1.0)
            X = X + V
            X = np.clip(X, lb, ub)

            particle_trajectories.append(X.copy())

            # Evaluate fitness
            current_vals = rosenbrock(X)

            # Update personal bests
            better_mask = current_vals < pbest_val
            pbest[better_mask] = X[better_mask]
            pbest_val[better_mask] = current_vals[better_mask]

            # Update global best
            if np.min(pbest_val) < gbest_val:
                gbest_val = np.min(pbest_val)
                gbest = pbest[np.argmin(pbest_val)].copy()

            history_gbest.append(gbest_val)

        return gbest, gbest_val, history_gbest, np.array(particle_trajectories)

def run_practical_3_8():
    print("=" * 60)
    print("PRACTICAL 3.8: Particle Swarm Optimization (PSO)")
    print("=" * 60)

    pso = ParticleSwarmOptimizer(n_particles=35, dim=2, bounds=(-2.0, 2.0), n_iter=60)
    gbest, gbest_val, history, trajectories = pso.optimize()

    print(f"Objective Function: 2D Rosenbrock Function (Banana Valley)")
    print(f"Theoretical Global Minimum at x* = (1.0, 1.0) with f(x*) = 0.0")
    print(f"\nPSO Found Minimum Position: ({gbest[0]:.5f}, {gbest[1]:.5f})")
    print(f"Best Cost Value f(gbest):  {gbest_val:.6f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Optimization Convergence Curve
    ax1.plot(range(len(history)), history, 'r-', linewidth=2.2)
    ax1.set_yscale('log')
    ax1.set_title("PSO Log-Scale Convergence Curve (Rosenbrock)", fontweight='bold')
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Global Best Cost (Log Scale)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Contour Plot with Swarm Trajectories
    x_span = np.linspace(-2.0, 2.0, 200)
    y_span = np.linspace(-1.0, 2.5, 200)
    XX, YY = np.meshgrid(x_span, y_span)
    grid = np.stack([XX, YY], axis=-1)
    ZZ = rosenbrock(grid)

    contour = ax2.contour(XX, YY, ZZ, levels=np.logspace(-1, 3, 20), cmap='viridis')
    ax2.scatter(trajectories[0, :, 0], trajectories[0, :, 1], color='blue', alpha=0.5, s=30, label='Initial Swarm (t=0)')
    ax2.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], color='orange', alpha=0.8, s=40, label='Final Swarm (t=60)')
    ax2.scatter(1.0, 1.0, color='red', marker='*', s=200, edgecolors='k', zorder=10, label='True Global Minimum (1, 1)')
    ax2.set_title("Swarm Migration Paths Across Rosenbrock Valley", fontweight='bold')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex08_particle_swarm_optimization.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.8 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_8()
