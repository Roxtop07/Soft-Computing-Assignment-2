"""
Practical 1.7: Mamdani Fuzzy Inference System (FIS)
Course: Soft Computing (EL1) - STDA2102 | Module 1: Fuzzy Logic and Systems

Aim:
To implement a simple Mamdani Fuzzy Inference System using Python and scikit-fuzzy.

Theory:
A Mamdani Fuzzy Inference System (Ebrahim Mamdani, 1975) maps crisp inputs to crisp outputs
via four systematic stages:
1. Fuzzification: Mapping crisp inputs into linguistic membership values.
2. Rule Base Evaluation: Computing firing strength α_i = min(μ_Ai(x), μ_Bi(y)).
3. Aggregation: Combining all clipped rule consequences via max operator:
       μ_agg(z) = max_i [ min(α_i, μ_Ci(z)) ]
4. Defuzzification: Converting aggregated fuzzy set into a crisp output via Centroid (COG):
       z* = ∫ z·μ_agg(z) dz / ∫ μ_agg(z) dz
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a + 1e-12), (c - x)/(c - b + 1e-12)), 0.0)

def run_mamdani_custom(food_quality, service_rating):
    """Fallback / Pure-Python Mamdani implementation (Tipping Problem)"""
    tip_u = np.linspace(0, 25, 500)
    # Output MFs for Tip: Low (0, 0, 12), Medium (0, 12.5, 25), High (12.5, 25, 25)
    tip_low = trimf(tip_u, 0, 0, 12.5)
    tip_med = trimf(tip_u, 0, 12.5, 25)
    tip_high = trimf(tip_u, 12.5, 25, 25)

    # Input evaluations: Quality [0, 10], Service [0, 10]
    # Food: Bad (0,0,5), Decent (0,5,10), Delicious (5,10,10)
    food_u = np.linspace(0, 10, 100)
    f_bad = float(np.interp(food_quality, food_u, trimf(food_u, 0, 0, 5)))
    f_dec = float(np.interp(food_quality, food_u, trimf(food_u, 0, 5, 10)))
    f_del = float(np.interp(food_quality, food_u, trimf(food_u, 5, 10, 10)))

    # Service: Poor (0,0,5), Acceptable (0,5,10), Amazing (5,10,10)
    serv_u = np.linspace(0, 10, 100)
    s_poor = float(np.interp(service_rating, serv_u, trimf(serv_u, 0, 0, 5)))
    s_acc = float(np.interp(service_rating, serv_u, trimf(serv_u, 0, 5, 10)))
    s_ama = float(np.interp(service_rating, serv_u, trimf(serv_u, 5, 10, 10)))

    # Rule 1: IF food is bad OR service is poor THEN tip is low
    r1 = max(f_bad, s_poor)
    # Rule 2: IF service is acceptable THEN tip is medium
    r2 = s_acc
    # Rule 3: IF food is delicious OR service is amazing THEN tip is high
    r3 = max(f_del, s_ama)

    # Mamdani implication & aggregation
    c1 = np.minimum(r1, tip_low)
    c2 = np.minimum(r2, tip_med)
    c3 = np.minimum(r3, tip_high)
    aggregated = np.maximum(c1, np.maximum(c2, c3))

    # Centroid defuzzification
    denom = np.sum(aggregated)
    tip_val = float(np.sum(tip_u * aggregated) / denom) if denom > 0 else 12.5
    return tip_val, tip_u, aggregated

def run_practical_7():
    print("=" * 60)
    print("PRACTICAL 1.7: Mamdani Fuzzy Inference System (Tipping Problem)")
    print("=" * 60)
    print(f"scikit-fuzzy installed: {HAS_SKFUZZY}")

    food_input = 6.5
    service_input = 9.2

    tip_val, tip_u, aggregated = run_mamdani_custom(food_input, service_input)

    print(f"\nInputs: Food Quality = {food_input}/10, Service Rating = {service_input}/10")
    print(f"-> Inferred Recommended Tip: {tip_val:.2f}%")

    # Visualization of Aggregated Output & Defuzzification
    plt.figure(figsize=(9, 5))
    plt.plot(tip_u, aggregated, 'b-', linewidth=2.5, label='Aggregated Fuzzy Output μ(z)')
    plt.fill_between(tip_u, aggregated, color='skyblue', alpha=0.3)
    plt.axvline(tip_val, color='red', linestyle='--', linewidth=2.2, label=f'Centroid Tip: {tip_val:.2f}%')
    plt.title("Mamdani FIS: Output Fuzzy Aggregation & Centroid Defuzzification", fontsize=12, fontweight='bold')
    plt.xlabel("Tip Percentage (%)", fontsize=11)
    plt.ylabel("Degree of Membership", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', framealpha=0.95)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod1_ex07_mamdani_inference.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 1.7 completed successfully.\n")

if __name__ == "__main__":
    run_practical_7()
