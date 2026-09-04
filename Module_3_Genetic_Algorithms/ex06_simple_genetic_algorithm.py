"""
Practical 3.6: Simple Genetic Algorithm (SGA) Implementation
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To create a simple Genetic Algorithm by combining selection, crossover, and mutation operations using Python.

Theory:
The Simple Genetic Algorithm (John Holland, 1975; David Goldberg, 1989) iteratively evolves a population
towards optimal regions of the search space via:
1. Population Initialization: Generating random binary/real chromosomes.
2. Fitness Evaluation: Assigning fitness to each chromosome.
3. Elitism: Preserving the single best individual directly to the next generation without alteration.
4. Selection: Mating pool construction via Tournament or Roulette Wheel selection.
5. Recombination (Crossover): Pairing parents to generate novel gene combinations with probability P_c.
6. Mutation: Introducing random allele perturbations with probability P_m.
7. Generational Replacement.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Multimodal Target Function to Maximize: f(x) = x * sin(10*pi*x) + 2.0 on [-1.0, 2.0]
def objective_function(x):
    return x * np.sin(10 * np.pi * x) + 2.0

def decode_chromosome(bitstring, a=-1.0, b=2.0):
    """Decodes binary string of length L into continuous value x in [a, b]."""
    decimal = int("".join(map(str, bitstring)), 2)
    max_decimal = (2 ** len(bitstring)) - 1
    return a + decimal * (b - a) / max_decimal

class SimpleGeneticAlgorithm:
    def __init__(self, pop_size=50, chrom_len=20, n_gen=60, pc=0.8, pm=0.03):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.n_gen = n_gen
        self.pc = pc
        self.pm = pm
        self.best_history = []
        self.avg_history = []
        self.best_individual = None
        self.best_fitness = -np.inf

    def evolve(self):
        np.random.seed(42)
        # 1. Initialize Population
        pop = np.random.randint(0, 2, size=(self.pop_size, self.chrom_len))

        for gen in range(self.n_gen):
            # Decode & Evaluate
            decoded_vals = np.array([decode_chromosome(ind) for ind in pop])
            fitnesses = objective_function(decoded_vals)

            # Track Statistics
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fit = fitnesses[gen_best_idx]
            gen_avg_fit = np.mean(fitnesses)

            self.best_history.append(gen_best_fit)
            self.avg_history.append(gen_avg_fit)

            if gen_best_fit > self.best_fitness:
                self.best_fitness = gen_best_fit
                self.best_individual = pop[gen_best_idx].copy()
                self.best_x = decoded_vals[gen_best_idx]

            # 2. Elitism: Keep best individual
            new_pop = [pop[gen_best_idx].copy()]

            # 3. Selection, Crossover, and Mutation loop
            while len(new_pop) < self.pop_size:
                # Tournament Selection (k=3)
                p1_idx = np.random.choice(self.pop_size, size=3)
                p1 = pop[p1_idx[np.argmax(fitnesses[p1_idx])]]

                p2_idx = np.random.choice(self.pop_size, size=3)
                p2 = pop[p2_idx[np.argmax(fitnesses[p2_idx])]]

                # Crossover
                if np.random.rand() < self.pc:
                    k = np.random.randint(1, self.chrom_len)
                    c1 = np.concatenate([p1[:k], p2[k:]])
                    c2 = np.concatenate([p2[:k], p1[k:]])
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                for child in [c1, c2]:
                    flips = np.random.rand(self.chrom_len) < self.pm
                    child[flips] = 1 - child[flips]
                    if len(new_pop) < self.pop_size:
                        new_pop.append(child)

            pop = np.array(new_pop)

        return self.best_x, self.best_fitness

def run_practical_3_6():
    print("=" * 60)
    print("PRACTICAL 3.6: Simple Genetic Algorithm (SGA)")
    print("=" * 60)

    sga = SimpleGeneticAlgorithm(pop_size=60, chrom_len=22, n_gen=50, pc=0.85, pm=0.02)
    best_x, best_fit = sga.evolve()

    print(f"Target Problem: Maximize f(x) = x * sin(10*pi*x) + 2.0 on [-1.0, 2.0]")
    print(f"\nOptimization Results after {sga.n_gen} Generations:")
    print(f"Optimal Parameter x* = {best_x:.6f}")
    print(f"Maximum Fitness f(x*) = {best_fit:.6f}")
    print(f"Theoretical Global Maximum ≈ 3.8502 at x ≈ 1.8506")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Optimization Curve
    generations = range(1, sga.n_gen + 1)
    ax1.plot(generations, sga.best_history, 'g-', linewidth=2.2, label='Best Fitness')
    ax1.plot(generations, sga.avg_history, 'b--', linewidth=1.8, label='Average Population Fitness')
    ax1.axhline(3.8502, color='red', linestyle=':', label='Global Optimum (3.8502)')
    ax1.set_title("SGA Generational Convergence Profile", fontweight='bold')
    ax1.set_xlabel("Generation Number")
    ax1.set_ylabel("Fitness f(x)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Plot 2: Fitness Landscape & Optimal Solution Discovered
    x_plot = np.linspace(-1.0, 2.0, 1000)
    y_plot = objective_function(x_plot)
    ax2.plot(x_plot, y_plot, 'navy', linewidth=1.5, label='Objective Landscape f(x)')
    ax2.scatter(best_x, best_fit, color='red', s=180, marker='*', zorder=5,
                label=f'SGA Best: x*={best_x:.3f}, f={best_fit:.3f}')
    ax2.set_title("Target Function Landscape & Discovered Optima", fontweight='bold')
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex06_simple_genetic_algorithm.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.6 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_6()
