"""
Practical 3.10: Complete Optimization System using DEAP Library
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To design and implement a complete optimization system using a Genetic Algorithm in Python
with the DEAP library by integrating population initialization, fitness evaluation,
selection, crossover, mutation, and convergence analysis.

Theory:
DEAP (Distributed Evolutionary Algorithms in Python) provides a modular, object-oriented framework:
1. creator: Dynamically constructs Fitness classes (e.g., FitnessMin with negative weights)
   and Individual classes inheriting from standard container types (list/array).
2. Toolbox: A functional registry storing problem-specific genetic operators:
   - Initialization: tools.initRepeat for population instantiation
   - Evaluation: Problem-specific fitness mapping
   - Selection: tools.selTournament (or selRoulette, selNSGA2)
   - Crossover: tools.cxBlend (BLX-α) or tools.cxTwoPoint
   - Mutation: tools.mutGaussian (with specified μ, σ, and per-gene probability indpb)
3. HallOfFame (HoF): Pareto-optimal archive preserving the absolute global elite individuals.
4. Statistics & Logbook: Tracks generational evolution metrics (min, mean, max, std).
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Problem Definition: 5-Dimensional Griewank Benchmark Function (Minimization)
# Global minimum: f(0, 0, ..., 0) = 0
def griewank_eval(individual):
    x = np.array(individual)
    d = len(x)
    sum_part = np.sum(x**2) / 4000.0
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return (sum_part - prod_part + 1.0,)

def setup_deap_system(dimensions=5, bounds=(-10.0, 10.0)):
    # 1. Setup Fitness and Individual Types
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 2. Attribute and Population Registration
    toolbox.register("attr_float", random.uniform, bounds[0], bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 3. Evolutionary Operators Registration
    toolbox.register("evaluate", griewank_eval)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def run_practical_3_10():
    print("=" * 60)
    print("PRACTICAL 3.10: Complete Optimization System using DEAP")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    dim = 5
    toolbox = setup_deap_system(dimensions=dim, bounds=(-10.0, 10.0))

    # Population & Evolutionary Parameters
    pop_size = 100
    n_gen = 60
    cx_prob = 0.85
    mut_prob = 0.25

    population = toolbox.population(n=pop_size)

    # Hall of Fame (Elite individual archive)
    hof = tools.HallOfFame(1)

    # Statistics & Logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    # Execute Canonical Evolutionary Algorithm (eaSimple)
    print(f"Evolving {pop_size} individuals over {n_gen} generations for 5D Griewank optimization...")
    pop, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cx_prob, mutpb=mut_prob,
        ngen=n_gen, stats=stats, halloffame=hof,
        verbose=False
    )

    best_ind = hof[0]
    best_cost = best_ind.fitness.values[0]

    print("\n--- DEAP Evolutionary Optimization Completed ---")
    print(f"Optimal Vector x* = {[round(val, 5) for val in best_ind]}")
    print(f"Minimum Griewank Cost f(x*) = {best_cost:.8f}")
    print(f"Target Global Optimum at Origin = 0.00000000")

    # Extract logbook stats
    gen = logbook.select("gen")
    min_fit = logbook.select("min")
    avg_fit = logbook.select("avg")
    std_fit = logbook.select("std")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Min and Mean Fitness Curves
    ax1.plot(gen, min_fit, 'g-', linewidth=2.5, label='Best (Min) Fitness')
    ax1.plot(gen, avg_fit, 'b--', linewidth=2.0, label='Average Fitness')
    ax1.set_title("DEAP Evolutionary Optimization: Griewank Function", fontweight='bold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Cost Value")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Plot 2: Fitness Standard Deviation (Diversity Metric)
    ax2.plot(gen, std_fit, 'purple', linewidth=2.2, label='Fitness Std Dev (Diversity)')
    ax2.set_title("Generational Standard Deviation Dynamics", fontweight='bold')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Standard Deviation (σ)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex10_deap_optimization.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.10 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_10()
