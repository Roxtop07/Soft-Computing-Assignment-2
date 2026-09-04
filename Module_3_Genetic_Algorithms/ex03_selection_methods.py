"""
Practical 3.3: Selection Operators in Genetic Algorithms
Course: Soft Computing (EL1) - STDA2102 | Module 3: Genetic Algorithms

Aim:
To implement basic selection methods for selecting suitable individuals from a population using Python.

Theory:
Selection operators model the Darwinian principle of "survival of the fittest".
1. Roulette Wheel Selection (Fitness Proportionate):
   Each individual i is assigned a selection probability:
       P(i) = f_i / ∑_{j=1}^N f_j
   Individuals with higher fitness occupy larger sectors on the wheel.
2. Tournament Selection:
   k individuals are chosen at random from the population; the one with highest fitness wins.
   Tournament size k controls selection pressure (larger k = higher exploitation).
3. Rank Selection:
   Individuals are sorted in ascending order of fitness and assigned ranks 1 to N.
   Selection probability is proportional to rank, eliminating scaling stagnation and premature convergence.
4. Stochastic Universal Sampling (SUS):
   Uses a single spin with N equally spaced pointers, providing zero-bias and minimal spread.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def roulette_wheel_selection(population, fitnesses, num_select=10):
    probs = fitnesses / np.sum(fitnesses)
    indices = np.random.choice(len(population), size=num_select, p=probs)
    return population[indices], indices

def tournament_selection(population, fitnesses, num_select=10, k=3):
    selected_indices = []
    for _ in range(num_select):
        contenders = np.random.choice(len(population), size=k, replace=False)
        winner = contenders[np.argmax(fitnesses[contenders])]
        selected_indices.append(winner)
    return population[selected_indices], np.array(selected_indices)

def rank_selection(population, fitnesses, num_select=10):
    sorted_idx = np.argsort(fitnesses) # lowest to highest
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, len(fitnesses) + 1)
    probs = ranks / np.sum(ranks)
    indices = np.random.choice(len(population), size=num_select, p=probs)
    return population[indices], indices

def sus_selection(population, fitnesses, num_select=10):
    total_fit = np.sum(fitnesses)
    step = total_fit / num_select
    start = np.random.uniform(0, step)
    pointers = [start + i * step for i in range(num_select)]

    cum_fit = np.cumsum(fitnesses)
    selected_indices = []
    for p in pointers:
        idx = np.searchsorted(cum_fit, p)
        selected_indices.append(min(idx, len(population) - 1))
    return population[selected_indices], np.array(selected_indices)

def run_practical_3_3():
    print("=" * 60)
    print("PRACTICAL 3.3: Selection Operators Comparison")
    print("=" * 60)

    # Population of 8 individuals with varying fitness
    pop_size = 8
    population = np.array([f"Ind_{i+1}" for i in range(pop_size)])
    fitnesses = np.array([12.0, 45.0, 8.0, 95.0, 22.0, 68.0, 15.0, 110.0])

    print("Initial Population & Fitness Values:")
    for ind, fit in zip(population, fitnesses):
        print(f"  {ind}: Fitness = {fit:6.1f} ({fit/np.sum(fitnesses)*100:5.2f}%)")

    # Run 1000 sampling trials to compare theoretical vs empirical selection frequency
    n_trials = 2000
    _, rw_idx = roulette_wheel_selection(population, fitnesses, n_trials)
    _, tour_idx = tournament_selection(population, fitnesses, n_trials, k=3)
    _, rank_idx = rank_selection(population, fitnesses, n_trials)
    _, sus_idx = sus_selection(population, fitnesses, n_trials)

    rw_freq = np.bincount(rw_idx, minlength=pop_size) / n_trials
    tour_freq = np.bincount(tour_idx, minlength=pop_size) / n_trials
    rank_freq = np.bincount(rank_idx, minlength=pop_size) / n_trials
    sus_freq = np.bincount(sus_idx, minlength=pop_size) / n_trials

    print("\nEmpirical Selection Probability Across 2000 Draws:")
    print(f"{'Individual':^10} | {'True Fit %':^12} | {'Roulette %':^12} | {'Tournament %':^14} | {'Rank %':^10} | {'SUS %':^10}")
    print("-" * 75)
    for i in range(pop_size):
        tf = (fitnesses[i] / np.sum(fitnesses)) * 100.0
        print(f"{population[i]:^10} | {tf:11.2f}% | {rw_freq[i]*100:11.2f}% | {tour_freq[i]*100:13.2f}% | {rank_freq[i]*100:9.2f}% | {sus_freq[i]*100:9.2f}%")

    # Visualization
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_pos = np.arange(pop_size)
    width = 0.2

    ax.bar(x_pos - 1.5*width, rw_freq*100, width, label='Roulette Wheel', color='#3498db')
    ax.bar(x_pos - 0.5*width, tour_freq*100, width, label='Tournament (k=3)', color='#e74c3c')
    ax.bar(x_pos + 0.5*width, rank_freq*100, width, label='Rank Selection', color='#2ecc71')
    ax.bar(x_pos + 1.5*width, sus_freq*100, width, label='SUS Selection', color='#9b59b6')

    ax.set_title("Comparison of Selection Pressure Across Evolutionary Operators", fontsize=12, fontweight='bold')
    ax.set_xlabel("Individual ID", fontsize=11)
    ax.set_ylabel("Selection Frequency (%)", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{ind}\n(fit={fit:.0f})" for ind, fit in zip(population, fitnesses)])
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod3_ex03_selection_methods.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 3.3 completed successfully.\n")

if __name__ == "__main__":
    run_practical_3_3()
