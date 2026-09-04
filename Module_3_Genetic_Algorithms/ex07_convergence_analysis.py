"""
Practical 3.7: Convergence Analysis and Diversity Tracking in Genetic Algorithms
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To study the convergence of a Genetic Algorithm by observing changes in fitness values over multiple generations using Python.

Theory:
Convergence analysis in Evolutionary Computation examines the search dynamics across generational time:
1. Best Fitness Curve: Tracks highest achieved adaptation (monotonically non-decreasing under elitism).
2. Mean Fitness Curve: Measures overall population quality improvement.
3. Worst Fitness: Captures exploratory excursions driven by mutation.
4. Population Diversity: The standard deviation of the population's genes:
       Diversity(t) = (1/N) * ∑ ||x_i(t) - x_mean(t)||
   High diversity early represents broad exploration; diminishing diversity reflects exploitation/convergence.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2D Sphere Function Optimization (Minimization -> Maximization: f(x) = 100 - (x1^2 + x2^2))
def fitness_function(X):
    return 100.0 - np.sum(X ** 2, axis=-1)

def run_ga_tracking(pop_size=60, n_gen=50, pm=0.05, k_tournament=3):
    np.random.seed(42)
    dim = 2
    # Initialize real-valued population in [-5, 5]
    pop = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))

    best_history = []
    mean_history = []
    worst_history = []
    diversity_history = []

    for gen in range(n_gen):
        fits = fitness_function(pop)

        best_history.append(np.max(fits))
        mean_history.append(np.mean(fits))
        worst_history.append(np.min(fits))

        # Diversity: Mean Euclidean distance to population centroid
        centroid = np.mean(pop, axis=0)
        diversity = np.mean(np.linalg.norm(pop - centroid, axis=1))
        diversity_history.append(diversity)

        # Elitism
        best_ind = pop[np.argmax(fits)].copy()
        new_pop = [best_ind]

        # Selection & Variation
        while len(new_pop) < pop_size:
            # Tournament selection
            c1 = np.random.choice(pop_size, size=k_tournament)
            p1 = pop[c1[np.argmax(fits[c1])]]
            c2 = np.random.choice(pop_size, size=k_tournament)
            p2 = pop[c2[np.argmax(fits[c2])]]

            # Arithmetic Crossover
            alpha = np.random.rand()
            child = alpha * p1 + (1.0 - alpha) * p2

            # Gaussian Mutation
            if np.random.rand() < pm:
                child += np.random.normal(0, 0.3, size=dim)
            child = np.clip(child, -5.0, 5.0)
            new_pop.append(child)

        pop = np.array(new_pop)

    return {
        'best': np.array(best_history),
        'mean': np.array(mean_history),
        'worst': np.array(worst_history),
        'diversity': np.array(diversity_history)
    }

def run_practical_3_7():
    print("=" * 60)
    print("PRACTICAL 3.7: Genetic Algorithm Convergence & Diversity Analysis")
    print("=" * 60)

    stats = run_ga_tracking(pop_size=60, n_gen=50, pm=0.08, k_tournament=3)

    print("\nConvergence History Sample (Generations 1, 10, 25, 50):")
    print(f"{'Generation':^12} | {'Best Fitness':^14} | {'Mean Fitness':^14} | {'Diversity':^12}")
    print("-" * 60)
    for g in [0, 9, 24, 49]:
        print(f"Gen {g+1:2d}       | {stats['best'][g]:14.4f} | {stats['mean'][g]:14.4f} | {stats['diversity'][g]:12.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    generations = range(1, 51)

    # Plot 1: Fitness Convergence Envelope
    ax1.plot(generations, stats['best'], 'g-', linewidth=2.5, label='Best Individual Fitness')
    ax1.plot(generations, stats['mean'], 'b-', linewidth=2.0, label='Mean Population Fitness')
    ax1.plot(generations, stats['worst'], 'r:', linewidth=1.5, label='Worst Individual Fitness')
    ax1.fill_between(generations, stats['worst'], stats['best'], color='lightblue', alpha=0.3, label='Population Fitness Spread')
    ax1.set_title("Fitness Convergence Profiles Over 50 Generations", fontweight='bold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness Score")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')

    # Plot 2: Population Genetic Diversity Dynamics
    ax2.plot(generations, stats['diversity'], color='purple', linewidth=2.5, label='Centroid Distance Diversity')
    ax2.set_title("Population Diversity Decay Curve (Convergence Indicator)", fontweight='bold')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Euclidean Distance to Centroid")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex07_convergence_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.7 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_7()
