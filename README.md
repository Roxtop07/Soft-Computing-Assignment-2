# Soft Computing (EL1) - Assignment 2: Practical Lab Record & Exercises
**Course Code:** STDA2102 | **Credits:** 4 (3-0-2) | **Term:** Fall/Spring 2026  
**Student Name:** Manish Kumar  
**Evaluation Scheme:** 100 Marks (30 Practicals Across 3 Core Modules)

---

## 📌 Executive Summary

This repository contains the complete, verified academic submission for **Assignment 2** of the Soft Computing course. It implements all **30 practical exercises** spanning Fuzzy Systems, Artificial Neural Networks, and Genetic Algorithms, complete with:
- **30 Standalone Executable Python Scripts** organized by module.
- **30 High-Resolution Output Visualizations** in [`output_plots/`](output_plots/).
- **4 Pre-Rendered Jupyter Notebooks** (1 consolidated master + 3 module notebooks) with all code executions and figures pre-cached.
- **Publication-Grade Lab Manual Record** in Word ([`.docx`](Assignment_2_Lab_Manual_Record.docx), 10.3 MB) and Markdown ([`.md`](Assignment_2_Lab_Manual_Record.md)) format with complete Aim, Theory, Source Code, Outputs, and Analytical Observations.

---

## 📂 Key Deliverables

| Deliverable | File / Directory | Description |
|---|---|---|
| **Academic Lab Manual (Word)** | [`Assignment_2_Lab_Manual_Record.docx`](Assignment_2_Lab_Manual_Record.docx) | 10.3 MB complete formal lab record with all 30 practicals, aims, theory, outputs, and embedded plots. |
| **Academic Lab Manual (Markdown)** | [`Assignment_2_Lab_Manual_Record.md`](Assignment_2_Lab_Manual_Record.md) | Markdown source specification of the lab manual. |
| **Master Jupyter Notebook** | [`Soft_Computing_Assignment_2_All_Practicals.ipynb`](Soft_Computing_Assignment_2_All_Practicals.ipynb) | 4.38 MB all-in-one executed notebook containing all 30 practicals with pre-rendered charts. |
| **Module 1 Notebook (Fuzzy)** | [`Module_1_Fuzzy_Logic.ipynb`](Module_1_Fuzzy_Logic.ipynb) | Pre-rendered notebook for Practicals 1.1 to 1.10 (1.34 MB). |
| **Module 2 Notebook (ANN)** | [`Module_2_Neural_Networks.ipynb`](Module_2_Neural_Networks.ipynb) | Pre-rendered notebook for Practicals 2.1 to 2.10 (1.29 MB). |
| **Module 3 Notebook (GA)** | [`Module_3_Genetic_Algorithms.ipynb`](Module_3_Genetic_Algorithms.ipynb) | Pre-rendered notebook for Practicals 3.1 to 3.10 (1.78 MB). |
| **Output Figures** | [`output_plots/`](output_plots/) | 30 high-DPI `.png` plots corresponding to each practical exercise. |
| **Automated Verification Suite** | [`verify_assignment_2.py`](verify_assignment_2.py) | Comprehensive test script verifying all 30 scripts, plots, and notebooks. |
| **Lab Manual Generator** | [`generate_lab_manual.py`](generate_lab_manual.py) | Compiles all practical codes and plots into the Word lab manual. |
| **Notebook Build Pipeline** | [`build_all_module_notebooks.py`](build_all_module_notebooks.py) / [`build_jupyter_notebook.py`](build_jupyter_notebook.py) | Notebook compilation and execution pipelines using `nbclient`. |
| **Syllabus Specification** | [`Soft_Computing_Course_Document.docx`](Soft_Computing_Course_Document.docx) | Official course document and syllabus outline. |

---

## 🧪 Comprehensive Index of 30 Practical Exercises

### Module 1: Fuzzy Logic and Systems (Practicals 1.1 – 1.10)
| # | Practical Script | Core Topic / Aim | Output Plot |
|:---:|---|---|---|
| **1.1** | [`ex01_simple_fuzzy_sets.py`](Module_1_Fuzzy_Logic/ex01_simple_fuzzy_sets.py) | Discrete membership representation (Young, Middle-Aged, Old) | `mod1_ex01_simple_fuzzy_sets.png` |
| **1.2** | [`ex02_membership_functions.py`](Module_1_Fuzzy_Logic/ex02_membership_functions.py) | Triangular, Trapezoidal, and Gaussian membership functions | `mod1_ex02_membership_functions.png` |
| **1.3** | [`ex03_fuzzy_set_operations.py`](Module_1_Fuzzy_Logic/ex03_fuzzy_set_operations.py) | Standard Zadeh & Algebraic Union, Intersection, Complement | `mod1_ex03_fuzzy_operations.png` |
| **1.4** | [`ex04_fuzzy_relations.py`](Module_1_Fuzzy_Logic/ex04_fuzzy_relations.py) | Cartesian product relation and Max-Min composition matrix | `mod1_ex04_fuzzy_relations.png` |
| **1.5** | [`ex05_linguistic_variables.py`](Module_1_Fuzzy_Logic/ex05_linguistic_variables.py) | Linguistic hedges (Very, Somewhat) and temperature scale | `mod1_ex05_linguistic_variables.png` |
| **1.6** | [`ex06_fuzzy_if_then_rules.py`](Module_1_Fuzzy_Logic/ex06_fuzzy_if_then_rules.py) | Multi-rule antecedent evaluation via min T-norm | `mod1_ex06_fuzzy_rules.png` |
| **1.7** | [`ex07_mamdani_inference.py`](Module_1_Fuzzy_Logic/ex07_mamdani_inference.py) | Mamdani Min-Max inference pipeline with aggregated output | `mod1_ex07_mamdani_inference.png` |
| **1.8** | [`ex08_sugeno_inference.py`](Module_1_Fuzzy_Logic/ex08_sugeno_inference.py) | Takagi-Sugeno-Kang zero/first-order linear inference | `mod1_ex08_sugeno_inference.png` |
| **1.9** | [`ex09_defuzzification_methods.py`](Module_1_Fuzzy_Logic/ex09_defuzzification_methods.py) | Centroid, Bisector, MOM, SOM, and LOM comparative analysis | `mod1_ex09_defuzzification.png` |
| **1.10** | [`ex10_fuzzy_logic_controller.py`](Module_1_Fuzzy_Logic/ex10_fuzzy_logic_controller.py) | Complete Fuzzy Controller (Tipping / Service system) | `mod1_ex10_fuzzy_controller.png` |

---

### Module 2: Artificial Neural Networks (Practicals 2.1 – 2.10)
| # | Practical Script | Core Topic / Aim | Output Plot |
|:---:|---|---|---|
| **2.1** | [`ex01_activation_functions.py`](Module_2_Neural_Networks/ex01_activation_functions.py) | Sigmoid, Tanh, ReLU, LeakyReLU, ELU & derivatives | `mod2_ex01_activation_functions.png` |
| **2.2** | [`ex02_mcculloch_pitts_neuron.py`](Module_2_Neural_Networks/ex02_mcculloch_pitts_neuron.py) | McCulloch-Pitts binary threshold model for AND & OR | `mod2_ex02_mcculloch_pitts.png` |
| **2.3** | [`ex03_perceptron_learning.py`](Module_2_Neural_Networks/ex03_perceptron_learning.py) | Rosenblatt Perceptron convergence with decision boundary | `mod2_ex03_perceptron_learning.png` |
| **2.4** | [`ex04_adaline_madaline.py`](Module_2_Neural_Networks/ex04_adaline_madaline.py) | Widrow-Hoff LMS gradient descent rule & MSE trajectory | `mod2_ex04_adaline_learning.png` |
| **2.5** | [`ex05_mlp_backpropagation.py`](Module_2_Neural_Networks/ex05_mlp_backpropagation.py) | Multi-Layer Perceptron trained on XOR with nonlinear surface | `mod2_ex05_mlp_backprop.png` |
| **2.6** | [`ex06_hebbian_learning.py`](Module_2_Neural_Networks/ex06_hebbian_learning.py) | Unsupervised Hebbian associative plasticity & weight vectors | `mod2_ex06_hebbian_learning.png` |
| **2.7** | [`ex07_hopfield_network.py`](Module_2_Neural_Networks/ex07_hopfield_network.py) | Auto-associative Hopfield memory with energy function convergence | `mod2_ex07_hopfield_network.png` |
| **2.8** | [`ex08_som_kohonen.py`](Module_2_Neural_Networks/ex08_som_kohonen.py) | Kohonen Self-Organizing Map topological weight grid | `mod2_ex08_som_kohonen.png` |
| **2.9** | [`ex09_radial_basis_function.py`](Module_2_Neural_Networks/ex09_radial_basis_function.py) | RBF Network function approximation with Gaussian kernels | `mod2_ex09_rbf_network.png` |
| **2.10** | [`ex10_deep_neural_network.py`](Module_2_Neural_Networks/ex10_deep_neural_network.py) | Deep ANN for classification with accuracy/loss curves | `mod2_ex10_deep_nn.png` |

---

### Module 3: Genetic Algorithms & Evolutionary Computing (Practicals 3.1 – 3.10)
| # | Practical Script | Core Topic / Aim | Output Plot |
|:---:|---|---|---|
| **3.1** | [`ex01_binary_chromosome_representation.py`](Module_3_Genetic_Algorithms/ex01_binary_chromosome_representation.py) | Binary string genotype-to-phenotype mapping | `mod3_ex01_binary_representation.png` |
| **3.2** | [`ex02_fitness_function_formulation.py`](Module_3_Genetic_Algorithms/ex02_fitness_function_formulation.py) | Multi-modal benchmark fitness landscape evaluation | `mod3_ex02_fitness_landscape.png` |
| **3.3** | [`ex03_selection_methods.py`](Module_3_Genetic_Algorithms/ex03_selection_methods.py) | Roulette Wheel, Tournament, and Rank selection comparisons | `mod3_ex03_selection_methods.png` |
| **3.4** | [`ex04_crossover_operators.py`](Module_3_Genetic_Algorithms/ex04_crossover_operators.py) | Single-point, Two-point, and Uniform crossover mechanics | `mod3_ex04_crossover_operators.png` |
| **3.5** | [`ex05_mutation_operators.py`](Module_3_Genetic_Algorithms/ex05_mutation_operators.py) | Bit-flip, Gaussian, and Inversion mutation diversity | `mod3_ex05_mutation_operators.png` |
| **3.6** | [`ex06_simple_genetic_algorithm.py`](Module_3_Genetic_Algorithms/ex06_simple_genetic_algorithm.py) | Canonical GA convergence on continuous optimization | `mod3_ex06_ga_convergence.png` |
| **3.7** | [`ex07_knapsack_problem_ga.py`](Module_3_Genetic_Algorithms/ex07_knapsack_problem_ga.py) | Combinatorial 0/1 Knapsack optimization with penalty function | `mod3_ex07_knapsack_ga.png` |
| **3.8** | [`ex08_tsp_order_crossover.py`](Module_3_Genetic_Algorithms/ex08_tsp_order_crossover.py) | Traveling Salesperson Problem (TSP) using Order Crossover (OX) | `mod3_ex08_tsp_ga.png` |
| **3.9** | [`ex09_real_coded_ga.py`](Module_3_Genetic_Algorithms/ex09_real_coded_ga.py) | Real-Coded GA with SBX and Polynomial Mutation | `mod3_ex09_real_coded_ga.png` |
| **3.10** | [`ex10_neuro_fuzzy_or_ga_nn.py`](Module_3_Genetic_Algorithms/ex10_neuro_fuzzy_or_ga_nn.py) | Neuro-Evolution: GA optimizing neural network weights | `mod3_ex10_hybrid_soft_computing.png` |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Create and activate Python 3.11 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all required scientific and ML dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Suite
Verify all 30 scripts, 30 plots, module notebooks, master notebook, and lab manuals in one command:
```bash
python verify_assignment_2.py
```

### 3. Run Individual Practical Exercises
Any script can be executed independently from the repository root:
```bash
# Example: Run Practical 1.10 (Fuzzy Controller)
python Module_1_Fuzzy_Logic/ex10_fuzzy_logic_controller.py

# Example: Run Practical 2.5 (MLP Backprop on XOR)
python Module_2_Neural_Networks/ex05_mlp_backpropagation.py

# Example: Run Practical 3.8 (TSP with GA)
python Module_3_Genetic_Algorithms/ex08_tsp_order_crossover.py
```

### 4. Regenerate Lab Manual Document (.docx & .md)
```bash
python generate_lab_manual.py
```

### 5. Re-execute Jupyter Notebooks
```bash
# Rebuild and execute master consolidated notebook
python build_jupyter_notebook.py

# Rebuild all 3 individual module notebooks
python build_all_module_notebooks.py
```
