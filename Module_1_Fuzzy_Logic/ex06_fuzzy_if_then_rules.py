"""
Practical 1.6: Fuzzy IF-THEN Rules and Implication
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To create simple Fuzzy IF–THEN rules for making decisions based on given input conditions.

Theory:
A Fuzzy IF-THEN rule has the canonical form:
    IF x is A AND y is B THEN z is C
where:
- "x is A AND y is B" is the Rule Antecedent (Premise)
- "z is C" is the Rule Consequent (Conclusion)

Firing Strength (Rule Weight α):
Evaluated using a T-norm (Min operator):
    α = min(μ_A(x*), μ_B(y*))

Fuzzy Implication (Mamdani Min Implication):
    μ_C'(z) = min(α, μ_C(z))
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

def run_practical_6():
    print("=" * 60)
    print("PRACTICAL 1.6: Fuzzy IF-THEN Rules Evaluation & Clipping")
    print("=" * 60)

    # Universes
    temp_u = np.linspace(10, 40, 300)
    hum_u = np.linspace(20, 90, 300)
    fan_u = np.linspace(0, 100, 300)

    # Antecedents
    temp_hot = trimf(temp_u, 25, 35, 40)
    hum_high = trimf(hum_u, 50, 75, 90)

    # Consequent
    fan_fast = trimf(fan_u, 50, 80, 100)

    # Given Crisp Conditions
    input_temp = 32.0  # °C
    input_hum = 70.0   # %

    # 1. Fuzzification
    mu_temp_val = float(np.interp(input_temp, temp_u, temp_hot))
    mu_hum_val = float(np.interp(input_hum, hum_u, hum_high))

    # 2. Rule: IF Temp is Hot AND Humidity is High THEN Fan is Fast
    firing_strength = min(mu_temp_val, mu_hum_val)

    # 3. Implication: Mamdani Minimum Clipping
    clipped_consequent = np.minimum(firing_strength, fan_fast)

    print(f"Input Condition: Temperature = {input_temp}°C, Humidity = {input_hum}%")
    print(f"  μ_Hot(Temp) = {mu_temp_val:.3f}")
    print(f"  μ_High(Hum) = {mu_hum_val:.3f}")
    print(f"Rule Firing Strength α = min({mu_temp_val:.3f}, {mu_hum_val:.3f}) = {firing_strength:.3f}")
    print(f"Max degree in clipped consequent: {np.max(clipped_consequent):.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Antecedent 1
    axes[0].plot(temp_u, temp_hot, 'r-', linewidth=2, label='Hot')
    axes[0].axvline(input_temp, color='black', linestyle='--', label=f'Input: {input_temp}°C')
    axes[0].plot(input_temp, mu_temp_val, 'ro', markersize=8)
    axes[0].set_title(f"Antecedent 1: Temp is Hot (μ = {mu_temp_val:.2f})", fontweight='bold')
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Membership")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Antecedent 2
    axes[1].plot(hum_u, hum_high, 'b-', linewidth=2, label='High')
    axes[1].axvline(input_hum, color='black', linestyle='--', label=f'Input: {input_hum}%')
    axes[1].plot(input_hum, mu_hum_val, 'bo', markersize=8)
    axes[1].set_title(f"Antecedent 2: Hum is High (μ = {mu_hum_val:.2f})", fontweight='bold')
    axes[1].set_xlabel("Humidity (%)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    # Consequent with Mamdani Clipping
    axes[2].plot(fan_u, fan_fast, 'g--', linewidth=1.5, label='Original Fan Fast')
    axes[2].plot(fan_u, clipped_consequent, 'g-', linewidth=2.5, label=f'Clipped (α = {firing_strength:.2f})')
    axes[2].fill_between(fan_u, clipped_consequent, color='green', alpha=0.25)
    axes[2].axhline(firing_strength, color='orange', linestyle=':', label='Firing Level α')
    axes[2].set_title("Consequent: Fan Fast (Clipped Output)", fontweight='bold')
    axes[2].set_xlabel("Fan Speed (%)")
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex06_fuzzy_rules.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.6 completed successfully.\n")

if __name__ == "__main__":
    run_practical_6()
