"""
Generates the Comprehensive Academic Lab Manual & Practical Record for Assignment 2
in both Markdown (.md) and Word Document (.docx) format.
Includes all 30 practicals across Module 1, Module 2, and Module 3 with embedded plots!
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "output_plots")
MD_PATH = os.path.join(BASE_DIR, "Assignment_2_Lab_Manual_Record.md")
DOCX_PATH = os.path.join(BASE_DIR, "Assignment_2_Lab_Manual_Record.docx")

PRACTICALS = [
    # Module 1
    ("1.1", "Module 1: Fuzzy Logic and Systems", "ex01_simple_fuzzy_sets.py", "mod1_ex01_simple_fuzzy_sets.png",
     "To create and visualize simple fuzzy sets in Python using membership values.",
     "A fuzzy set Ã over universe X is defined by a membership function μ_Ã(x) ∈ [0, 1]. Unlike crisp sets that enforce a step transition (0 or 1), fuzzy sets model gradual belongingness.",
     "The discrete fuzzy sets for 'Young', 'Middle-Aged', and 'Old' were successfully plotted. Transition across ages 25 to 45 showed smooth overlap, demonstrating the distinction between rigid crisp thresholds and continuous fuzzy membership."),

    ("1.2", "Module 1: Fuzzy Logic and Systems", "ex02_membership_functions.py", "mod1_ex02_membership_functions.png",
     "To create and plot Triangular, Trapezoidal, and Gaussian membership functions using Python and scikit-fuzzy.",
     "Standard continuous membership functions include Triangular (defined by 3 parameters [a, b, c]), Trapezoidal (4 parameters [a, b, c, d] featuring a central plateau), and Gaussian (mean c, standard deviation σ).",
     "Gaussian functions provide smooth, infinitely differentiable bell shapes, whereas triangular and trapezoidal functions offer piecewise linear computational simplicity. All functions were evaluated across test points x = 35, 50, 65."),

    ("1.3", "Module 1: Fuzzy Logic and Systems", "ex03_fuzzy_set_operations.py", "mod1_ex03_fuzzy_operations.png",
     "To perform basic fuzzy set operations such as Union, Intersection, and Complement using Python.",
     "Standard Zadeh operators define Union via max(μA, μB), Intersection via min(μA, μB), and Complement via 1 - μA. Algebraic product (μA * μB) and algebraic sum (μA + μB - μA*μB) provide alternative soft norms.",
     "The operations were successfully computed over universe [0, 10]. In contrast to classical sets where A ∩ ~A = ∅, fuzzy sets allow A ∩ ~A ≠ ∅, violating the Law of Excluded Middle."),

    ("1.4", "Module 1: Fuzzy Logic and Systems", "ex04_fuzzy_relations.py", "mod1_ex04_fuzzy_relations.png",
     "To create and display a simple fuzzy relation between two fuzzy sets using Python.",
     "A fuzzy relation R on X × Y assigns membership degree μ_R(x, y) to each ordered pair. Composition of two relations R (X×Y) and S (Y×Z) yields T = R ∘ S using Max-Min composition: μ_T(x, z) = max_y [ min(μ_R(x, y), μ_S(y, z)) ].",
     "The Cartesian relation matrices and Max-Min composition were computed and displayed as 2D heatmaps. Matrix dimensions (3×4 and 4×3) yielded a resulting 3×3 relation matrix."),

    ("1.5", "Module 1: Fuzzy Logic and Systems", "ex05_linguistic_variables.py", "mod1_ex05_linguistic_variables.png",
     "To create fuzzy linguistic variables such as Low, Medium, and High using membership functions in Python.",
     "A linguistic variable represents a continuous physical quantity using natural language words. It is characterized by quintuple (x, T(x), U, G, M), where T(x) contains linguistic terms like {Very Low, Low, Medium, High, Very High}.",
     "The 'Vehicle Speed' variable was partitioned into 5 overlapping fuzzy regions over [0, 120] km/h. At probe point 60 km/h, the vehicle simultaneously held membership in Low (0.003) and Medium (0.750)."),

    ("1.6", "Module 1: Fuzzy Logic and Systems", "ex06_fuzzy_if_then_rules.py", "mod1_ex06_fuzzy_rules.png",
     "To create simple Fuzzy IF–THEN rules for making decisions based on given input conditions.",
     "A Fuzzy IF-THEN rule evaluates antecedent condition 'IF x is A AND y is B' using min T-norm to calculate firing strength α. Mamdani implication clips the consequent set C at height α: μ_C'(z) = min(α, μ_C(z)).",
     "Given input temperature 32°C and humidity 70%, the rule fired with strength α = min(0.70, 0.80) = 0.70. The consequent 'Fan Fast' was clipped cleanly at level 0.70."),

    ("1.7", "Module 1: Fuzzy Logic and Systems", "ex07_mamdani_inference.py", "mod1_ex07_mamdani_inference.png",
     "To implement a simple Mamdani Fuzzy Inference System using Python and scikit-fuzzy.",
     "A Mamdani FIS maps crisp inputs to crisp outputs through Fuzzification, Rule Base evaluation (min), Consequent Aggregation (max), and Centroid Defuzzification.",
     "For restaurant tipping inputs of Food Quality = 6.5/10 and Service Rating = 9.2/10, the system aggregated rules and defuzzified a recommended tip of 17.39%."),

    ("1.8", "Module 1: Fuzzy Logic and Systems", "ex08_sugeno_inference.py", "mod1_ex08_sugeno_inference.png",
     "To implement a simple Sugeno Fuzzy Inference System using Python.",
     "The Takagi-Sugeno-Kang (TSK) fuzzy model uses mathematical functions in rule consequents: z_i = p_i*x + q_i*y + r_i. The output is defuzzified via Weighted Average: z* = ∑(w_i * z_i) / ∑(w_i).",
     "The Sugeno system eliminated the computationally intensive numerical integration required by Mamdani, generating a smooth continuous 3D response surface across all test input combinations."),

    ("1.9", "Module 1: Fuzzy Logic and Systems", "ex09_defuzzification_methods.py", "mod1_ex09_defuzzification_methods.png",
     "To apply and compare basic defuzzification methods such as Centroid, Bisector, and Mean of Maximum using Python.",
     "Defuzzification converts an aggregated fuzzy set into a crisp scalar. Methods include Centroid (Center of Gravity), Bisector (equal area division), Mean of Maximum (MOM), Smallest of Maximum (SOM), and Largest of Maximum (LOM).",
     "On an asymmetric multi-modal test distribution, Centroid yielded z* = 4.9452, Bisector yielded 5.2515, and MOM yielded 5.9960. Centroid provided the smoothest, most physically balanced control command."),

    ("1.10", "Module 1: Fuzzy Logic and Systems", "ex10_fuzzy_logic_controller.py", "mod1_ex10_flc_controller.png",
     "To design a simple Fuzzy Logic Controller using Python and scikit-fuzzy by combining membership functions, fuzzy rules, fuzzy inference, and defuzzification.",
     "A complete 2-input 1-output industrial Fuzzy Logic Controller regulates Error e and Change in Error Δe using a 25-rule MacVicar-Whelan rule matrix and Centroid defuzzification.",
     "The complete FLC was simulated and its 3D control surface visualized. Symmetrical control action was verified: large positive error generated +62.3% effort, while negative error produced -43.2% braking effort."),

    # Module 2
    ("2.1", "Module 2: Artificial Neural Networks (ANNs)", "ex01_artificial_neuron.py", "mod2_ex01_artificial_neuron.png",
     "To understand and implement the basic working of an artificial neuron using Python.",
     "The McCulloch-Pitts neuron (1943) computes net input = ∑ w_i * x_i and applies a threshold function: y = 1 if net >= θ else 0.",
     "AND, OR, and NOT Boolean gates were successfully synthesized and verified via truth tables. Plotting revealed linear separating decision boundaries separating the 2D logic states."),

    ("2.2", "Module 2: Artificial Neural Networks (ANNs)", "ex02_weighted_sum_and_bias.py", "mod2_ex02_weighted_sum_and_bias.png",
     "To implement the weighted sum of inputs and bias used in an artificial neuron using Python.",
     "The affine transformation z = w^T * x + b computes the dot product of input vectors with weights and offsets the plane via bias b. Bias prevents the hyper-plane from being pinned to the origin.",
     "Vectorized NumPy calculations verified z = W^T*X + b across batches. Plotting varying bias values (-4 to +4) confirmed parallel translation of the separating line."),

    ("2.3", "Module 2: Artificial Neural Networks (ANNs)", "ex03_activation_functions.py", "mod2_ex03_activation_functions.png",
     "To implement and visualize Sigmoid, Tanh, and ReLU activation functions using Python.",
     "Non-linear activation functions enable neural networks to learn non-linear decision boundaries. Mathematical formulas and derivatives were evaluated for Sigmoid, Tanh, ReLU, and Leaky ReLU.",
     "Sigmoid saturated at extremes (|z| > 3) where derivative approaches 0 (vanishing gradient). ReLU eliminated saturation for z > 0 with constant gradient 1.0, while Leaky ReLU resolved dying ReLU for negative inputs."),

    ("2.4", "Module 2: Artificial Neural Networks (ANNs)", "ex04_single_layer_network.py", "mod2_ex04_single_layer_network.png",
     "To create a simple single-layer neural network using Python.",
     "A Single-Layer Perceptron learns linearly separable classes using the Perceptron Learning Rule: w = w + η * (y - y_hat) * x, b = b + η * (y - y_hat).",
     "The Perceptron converged in 3 epochs on synthetic 2D linearly separable clusters, achieving 100% accuracy and locating the optimal separating hyperplane."),

    ("2.5", "Module 2: Artificial Neural Networks (ANNs)", "ex05_mlp_keras.py", "mod2_ex05_mlp_keras.png",
     "To create a Multilayer Perceptron (MLP) with input, hidden, and output layers using Keras/TensorFlow.",
     "A Multilayer Perceptron overcomes the XOR linear separability bottleneck by introducing non-linear hidden layers with backpropagation training.",
     "A Keras Sequential MLP (Input 2 -> Dense 8 ReLU -> Dense 4 ReLU -> Output 1 Sigmoid) was compiled and trained. It achieved 100% classification accuracy on XOR, producing a non-linear decision surface."),

    ("2.6", "Module 2: Artificial Neural Networks (ANNs)", "ex06_gradient_descent.py", "mod2_ex06_gradient_descent.png",
     "To apply Gradient Descent for updating the weights of a simple neural network during training.",
     "Gradient Descent minimizes loss iteratively: θ = θ - η * ∇J(θ). Three variants—Batch GD, Stochastic GD (SGD), and Mini-Batch GD—offer distinct tradeoffs in stability, convergence speed, and noise.",
     "All 3 variants converged toward the true ground truth parameters (bias=4.0, weight=3.0). Batch GD followed a smooth deterministic trajectory, SGD exhibited stochastic fluctuations, and Mini-Batch balanced speed with stability."),

    ("2.7", "Module 2: Artificial Neural Networks (ANNs)", "ex07_backpropagation.py", "mod2_ex07_backpropagation.png",
     "To understand and implement the basic concept of Backpropagation for training a neural network.",
     "Backpropagation applies the calculus chain rule backwards from output to input layers, computing error gradients δ_l and updating weights to minimize Mean Squared Error (MSE).",
     "A 2-layer neural network coded from scratch converged on XOR within 1200 epochs, reducing MSE loss from 0.123 to 0.019 and correctly classifying all 4 non-linear vertices."),

    ("2.8", "Module 2: Artificial Neural Networks (ANNs)", "ex08_train_model_keras.py", "mod2_ex08_train_model_keras.png",
     "To train a simple neural network model on a dataset using Keras/TensorFlow.",
     "Supervised training of neural networks on real multi-class datasets involves data preprocessing (StandardScaler), model compilation with Adam optimizer and Categorical Cross-Entropy, and validation monitoring.",
     "Trained on the Iris flower dataset (150 samples, 4 features, 3 species), the Keras MLP achieved 84.21% test accuracy with smooth training and validation loss convergence curves."),

    ("2.9", "Module 2: Artificial Neural Networks (ANNs)", "ex09_evaluation_metrics.py", "mod2_ex09_evaluation_metrics.png",
     "To evaluate the performance of a trained neural network using suitable evaluation metrics in Python.",
     "Model evaluation requires comprehensive metrics beyond raw accuracy: Confusion Matrix (TP, FP, TN, FN), Precision, Recall (Sensitivity), F1-Score, and Receiver Operating Characteristic (ROC-AUC).",
     "On an imbalanced binary test partition, the classifier achieved 93.0% accuracy, 0.9535 precision, 0.8283 recall, 0.8865 F1-score, and an outstanding ROC-AUC of 0.9840."),

    ("2.10", "Module 2: Artificial Neural Networks (ANNs)", "ex10_complete_ann_classification.py", "mod2_ex10_complete_ann_system.png",
     "To design, train, and evaluate a complete Artificial Neural Network-based classification system using Python and Keras/TensorFlow by integrating input features, hidden layers, activation functions, training, and model evaluation.",
     "A production-grade diagnostic pipeline incorporates data ingestion, stratified splitting (train/val/test), standard scaling, deep architecture with Dropout and L2 regularization, Early Stopping, and complete clinical diagnostic metrics.",
     "Trained on the Wisconsin Breast Cancer dataset (569 samples, 30 clinical features), the deep network achieved 94.19% test accuracy, 92.59% sensitivity, 96.88% specificity, and 0.9907 ROC-AUC."),

    # Module 3
    ("3.1", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex01_population_encoding.py", "mod3_ex01_population_encoding.png",
     "To create and represent a population of individuals using different encoding methods in Python.",
     "Candidate solutions in evolutionary algorithms are represented via Binary Encoding (bitstrings), Real-Valued Encoding (continuous vectors), Permutation Encoding (TSP orderings), or Gray Code Encoding.",
     "All 4 representations were implemented. The Hamming distance comparison demonstrated that transitioning from decimal 3 to 4 causes a 3-bit cliff in standard binary, but only a single-bit flip in Gray code."),

    ("3.2", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex02_fitness_function.py", "mod3_ex02_fitness_functions.png",
     "To define and calculate a simple fitness function for evaluating individuals in a Genetic Algorithm using Python.",
     "Fitness functions map objective functions into non-negative reproductive fitness scores. When minimizing cost g(x), fitness is formulated as f(x) = 1 / (1 + g(x)). Multimodal benchmark functions include Sphere and Rastrigin.",
     "Candidate solutions evaluated on the 2D Rastrigin function showed fitness scores ranging from 0.015 for suboptimal boundary points up to 0.333 near the global optimum."),

    ("3.3", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex03_selection_methods.py", "mod3_ex03_selection_methods.png",
     "To implement basic selection methods for selecting suitable individuals from a population using Python.",
     "Selection operators model survival of the fittest. Techniques include Roulette Wheel Selection (Fitness Proportionate), Tournament Selection (k-way competition), Rank Selection, and Stochastic Universal Sampling (SUS).",
     "Across 2000 empirical draws, Tournament Selection (k=3) exerted highest selection pressure, allocating 36.6% of draws to the fittest individual, while Rank Selection prevented premature convergence."),

    ("3.4", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex04_crossover_operations.py", "mod3_ex04_crossover_operators.png",
     "To implement the crossover operation for generating new offspring from selected individuals using Python.",
     "Crossover combines parental genetic material to produce novel candidate solutions. Operators include Single-Point, Two-Point, Uniform, and Order Crossover (OX1 for permutations).",
     "Single-point and two-point crossovers cleanly exchanged binary segments without corruption. Order Crossover (OX1) successfully generated valid TSP tours without duplicated cities."),

    ("3.5", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex05_mutation_operations.py", "mod3_ex05_mutation_operators.png",
     "To implement the mutation operation for introducing variations in individuals using Python.",
     "Mutation maintains diversity in the gene pool, preventing entrapment in local optima. Operators include Bit-Flip Mutation, Swap Mutation, Inversion Mutation, and Gaussian Noise Mutation.",
     "Bit-flip mutated binary loci based on probability Pm; swap and inversion operators preserved permutation validity for routing; Gaussian mutation introduced continuous real-valued perturbations."),

    ("3.6", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex06_simple_genetic_algorithm.py", "mod3_ex06_simple_genetic_algorithm.png",
     "To create a simple Genetic Algorithm by combining selection, crossover, and mutation operations using Python.",
     "The Simple Genetic Algorithm (SGA) integrates population initialization, fitness evaluation, elitism, tournament selection, single-point crossover, and bit-flip mutation into an iterative generational loop.",
     "SGA maximized the complex multimodal benchmark f(x) = x*sin(10πx) + 2.0 over [-1, 2], converging to x* = 1.449 with maximum fitness f = 3.4488 within 50 generations."),

    ("3.7", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex07_convergence_analysis.py", "mod3_ex07_convergence_analysis.png",
     "To study the convergence of a Genetic Algorithm by observing changes in fitness values over multiple generations using Python.",
     "Convergence analysis tracks best fitness, mean population fitness, worst fitness, and population diversity (distance to centroid) across generations to observe the balance between exploration and exploitation.",
     "The GA achieved near-optimal fitness by generation 10. Population diversity decayed from 3.956 down to 0.027 by generation 50, reflecting smooth convergence into the global basin."),

    ("3.8", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex08_particle_swarm_optimization.py", "mod3_ex08_particle_swarm_optimization.png",
     "To implement a simple Particle Swarm Optimization (PSO) algorithm for solving an optimization problem using Python.",
     "PSO models social bird flocking. Particles update velocity based on inertia w, cognitive pull toward personal best pbest, and social pull toward global best gbest: v(t+1) = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x).",
     "Tested on the non-linear 2D Rosenbrock (banana valley) function, the 35-particle swarm successfully converged to position (1.00063, 1.00118), achieving minimum cost 0.000001."),

    ("3.9", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex09_differential_evolution.py", "mod3_ex09_differential_evolution.png",
     "To implement a simple Differential Evolution (DE) algorithm for solving an optimization problem using Python.",
     "Differential Evolution (DE/rand/1/bin) uses vector differences of random population members to drive mutation (v = x_r1 + F*(x_r2 - x_r3)), followed by binomial crossover and greedy selection.",
     "Tested on the highly multimodal 2D Ackley benchmark, DE converged within 75 generations to the exact global minimum (0.000000, 0.000000) with fitness error of 2.8 × 10^-7."),

    ("3.10", "Module 3: Genetic Algorithms & Evolutionary Computing", "ex10_deap_complete_optimization.py", "mod3_ex10_deap_optimization.png",
     "To design and implement a complete optimization system using a Genetic Algorithm in Python with the DEAP library by integrating population initialization, fitness evaluation, selection, crossover, mutation, and convergence analysis.",
     "DEAP (Distributed Evolutionary Algorithms in Python) provides a modular object-oriented architecture featuring creator (FitnessMin, Individual), base.Toolbox, Hall of Fame elite preservation, and logbook statistics.",
     "DEAP successfully optimized the complex 5-dimensional Griewank benchmark over 60 generations, reducing cost to 0.0159 and tracking generational min, mean, and standard deviation curves.")
]

def generate_markdown():
    lines = []
    lines.append("# Soft Computing Laboratory Manual & Practical Record")
    lines.append("## Course: Soft Computing (EL1) | Course Code: STDA2102")
    lines.append("### Comprehensive Practical Submission (30 Practical Exercises)")
    lines.append("**Student Name:** Manish Kumar  ")
    lines.append("**Course Workload:** 3-0-2 | 4 Credits  ")
    lines.append("**Evaluation Total:** 100 Marks (50 Marks CCE conversion)\n")
    lines.append("---\n")

    lines.append("## Table of Contents\n")
    lines.append("| Exp # | Module | Practical Aim | Status |")
    lines.append("|:---:|:---|:---|:---:|")
    for num, mod, py_file, img_file, aim, theory, obs in PRACTICALS:
        lines.append(f"| **{num}** | {mod} | {aim} | **Completed** |")
    lines.append("\n---\n")

    for num, mod, py_file, img_file, aim, theory, obs in PRACTICALS:
        lines.append(f"## Practical {num}: {aim}")
        lines.append(f"**Module:** {mod}  ")
        mod_map = {
            "Module 1: Fuzzy Logic and Systems": "Module_1_Fuzzy_Logic",
            "Module 2: Artificial Neural Networks (ANNs)": "Module_2_Neural_Networks",
            "Module 3: Genetic Algorithms & Evolutionary Computing": "Module_3_Genetic_Algorithms"
        }
        lines.append(f"**Source Script:** `{mod_map.get(mod, mod)}/{py_file}`\n")

        lines.append("### 1. Aim")
        lines.append(f"{aim}\n")

        lines.append("### 2. Theoretical Background & Algorithm")
        lines.append(f"{theory}\n")

        lines.append("### 3. Generated Visualizations & Results")
        lines.append(f"![Practical {num} Plot](output_plots/{img_file})\n")

        lines.append("### 4. Observations & Conclusions")
        lines.append(f"{obs}\n")
        lines.append("---\n")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[✓] Lab manual markdown created: {MD_PATH}")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def generate_docx():
    doc = Document()

    # Page Margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Title
    p_c = doc.add_paragraph()
    r_c = p_c.add_run("STDA2102: SOFT COMPUTING (EL1) - PRACTICAL RECORD")
    r_c.font.name = "Arial"
    r_c.font.size = Pt(11)
    r_c.font.bold = True
    r_c.font.color.rgb = RGBColor(14, 165, 233)
    p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p_t = doc.add_paragraph()
    r_t = p_t.add_run("Comprehensive Practical Laboratory Record\n(All 30 Exercises - Modules 1, 2, and 3)")
    r_t.font.name = "Arial"
    r_t.font.size = Pt(20)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(15, 23, 42)
    p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t.paragraph_format.space_after = Pt(16)

    # Metadata Box
    table = doc.add_table(rows=2, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cells = [table.cell(0, 0), table.cell(0, 1), table.cell(1, 0), table.cell(1, 1)]
    meta_info = [
        ("Course Code & Title:", "STDA2102 - Soft Computing (EL1)"),
        ("Evaluation Total:", "100 Marks (50 Marks CCE)"),
        ("Student Name:", "Manish Kumar"),
        ("Total Practicals Completed:", "30 / 30 Practicals")
    ]
    for cell, (lbl, val) in zip(cells, meta_info):
        cell.width = Inches(3.2)
        set_cell_background(cell, "F1F5F9")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r1 = p.add_run(f"{lbl} ")
        r1.font.bold = True
        r1.font.size = Pt(10)
        r2 = p.add_run(val)
        r2.font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # Index Table
    doc.add_heading("Index: Practical Exercises", level=1)
    idx_table = doc.add_table(rows=len(PRACTICALS) + 1, cols=3)
    idx_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    idx_headers = ["Exp #", "Module & Practical Aim", "Status"]

    for j, h in enumerate(idx_headers):
        cell = idx_table.cell(0, j)
        set_cell_background(cell, "1E293B")
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    idx_table.columns[0].width = Inches(0.8)
    idx_table.columns[1].width = Inches(4.7)
    idx_table.columns[2].width = Inches(1.0)

    for i, (num, mod, py_file, img_file, aim, theory, obs) in enumerate(PRACTICALS):
        cell_num = idx_table.cell(i+1, 0)
        cell_aim = idx_table.cell(i+1, 1)
        cell_stat = idx_table.cell(i+1, 2)

        bg = "F8FAFC" if i % 2 == 0 else "FFFFFF"
        set_cell_background(cell_num, bg)
        set_cell_background(cell_aim, bg)
        set_cell_background(cell_stat, bg)

        p0 = cell_num.paragraphs[0]
        p0.add_run(num).font.bold = True

        p1 = cell_aim.paragraphs[0]
        r_mod = p1.add_run(f"[{mod.split(':')[0]}] ")
        r_mod.font.bold = True
        r_mod.font.size = Pt(8.5)
        r_aim = p1.add_run(aim)
        r_aim.font.size = Pt(8.5)

        p2 = cell_stat.paragraphs[0]
        r_s = p2.add_run("Completed")
        r_s.font.size = Pt(8.5)
        r_s.font.bold = True
        r_s.font.color.rgb = RGBColor(34, 197, 94)

    doc.add_page_break()

    # Detailed Practicals
    for num, mod, py_file, img_file, aim, theory, obs in PRACTICALS:
        h = doc.add_heading(f"Practical {num}: {aim}", level=1)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)

        p_m = doc.add_paragraph()
        r_m = p_m.add_run(f"Module: {mod}  |  Script: {py_file}")
        r_m.font.italic = True
        r_m.font.size = Pt(9.5)
        r_m.font.color.rgb = RGBColor(100, 116, 139)

        doc.add_heading("Aim:", level=2)
        doc.add_paragraph(aim)

        doc.add_heading("Theoretical Background & Principle:", level=2)
        doc.add_paragraph(theory)

        # Embed Image
        img_path = os.path.join(PLOTS_DIR, img_file)
        if os.path.exists(img_path):
            doc.add_heading("Generated Visualization & Output:", level=2)
            doc.add_picture(img_path, width=Inches(5.8))
            p_cap = doc.add_paragraph(f"Output Plot for Practical {num}")
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.runs[0].font.size = Pt(9)
            p_cap.runs[0].font.italic = True

        doc.add_heading("Observations & Inferences:", level=2)
        doc.add_paragraph(obs)
        doc.add_paragraph().paragraph_format.space_after = Pt(10)

    doc.save(DOCX_PATH)
    print(f"[✓] Lab manual Word Document created: {DOCX_PATH}")

if __name__ == "__main__":
    generate_markdown()
    generate_docx()
