# Soft Computing (EL1) - Assignment 2: Practical Lab Record & Exercises
**Course Code:** STDA2102 | **Credits:** 4 (3-0-2) | **Term:** Fall/Spring 2026  
**Student Name:** Manish Kumar  
**Evaluation Scheme:** 100 Marks (20 Practicals Across 2 Core Modules)  
**Repository:** [https://github.com/Roxtop07/Soft-Computing-Assignment-2](https://github.com/Roxtop07/Soft-Computing-Assignment-2)

---

## 📌 Executive Summary

This repository contains the complete, verified academic submission for **Assignment 2** of the Soft Computing course. It implements all **20 practical exercises** spanning **Module 1 (Fuzzy Logic and Systems)** and **Module 2 (Artificial Neural Networks)**, complete with:
- **20 Standalone Executable Python Scripts** organized by module.
- **20 High-Resolution Output Visualizations** in [`output_plots/`](output_plots/).
- **3 Pre-Rendered Jupyter Notebooks** (1 consolidated master + 2 module notebooks) with all code executions and figures pre-cached.
- **Publication-Grade Lab Manual Record** in Word ([`.docx`](Assignment_2_Lab_Manual_Record.docx), 5.72 MB) and Markdown ([`.md`](Assignment_2_Lab_Manual_Record.md)) format with complete Aim, Theory, Source Code, Outputs, and Analytical Observations.

---

## 📂 Key Deliverables

| Deliverable | File / Directory | Description |
|---|---|---|
| **Academic Lab Manual (Word)** | [`Assignment_2_Lab_Manual_Record.docx`](Assignment_2_Lab_Manual_Record.docx) | 5.72 MB complete formal lab record with all 20 practicals, aims, theory, outputs, and embedded plots. |
| **Academic Lab Manual (Markdown)** | [`Assignment_2_Lab_Manual_Record.md`](Assignment_2_Lab_Manual_Record.md) | Markdown source specification of the lab manual. |
| **Master Jupyter Notebook** | [`Soft_Computing_Assignment_2_All_Practicals.ipynb`](Soft_Computing_Assignment_2_All_Practicals.ipynb) | 2.62 MB all-in-one executed notebook containing all 20 practicals with pre-rendered charts. |
| **Module 1 Notebook (Fuzzy)** | [`Module_1_Fuzzy_Logic.ipynb`](Module_1_Fuzzy_Logic.ipynb) | Pre-rendered notebook for Practicals 1.1 to 1.10 (1.34 MB). |
| **Module 2 Notebook (ANN)** | [`Module_2_Neural_Networks.ipynb`](Module_2_Neural_Networks.ipynb) | Pre-rendered notebook for Practicals 2.1 to 2.10 (1.29 MB). |
| **Output Figures** | [`output_plots/`](output_plots/) | 20 high-DPI `.png` plots corresponding to each practical exercise. |
| **Automated Verification Suite** | [`verify_assignment_2.py`](verify_assignment_2.py) | Comprehensive test script verifying all 20 scripts, plots, and notebooks. |
| **Lab Manual Generator** | [`generate_lab_manual.py`](generate_lab_manual.py) | Compiles all practical codes and plots into the Word lab manual. |
| **Notebook Build Pipeline** | [`build_all_module_notebooks.py`](build_all_module_notebooks.py) / [`build_jupyter_notebook.py`](build_jupyter_notebook.py) | Notebook compilation and execution pipelines using `nbclient`. |
| **Syllabus Specification** | [`Soft_Computing_Course_Document.docx`](Soft_Computing_Course_Document.docx) | Official course document and syllabus outline. |

---

## 🧪 Comprehensive Index of 20 Practical Exercises

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
| **1.9** | [`ex09_defuzzification_methods.py`](Module_1_Fuzzy_Logic/ex09_defuzzification_methods.py) | Centroid, Bisector, MOM, SOM, and LOM comparative analysis | `mod1_ex09_defuzzification_methods.png` |
| **1.10** | [`ex10_fuzzy_logic_controller.py`](Module_1_Fuzzy_Logic/ex10_fuzzy_logic_controller.py) | Complete Fuzzy Controller (Tipping / Service system) | `mod1_ex10_flc_controller.png` |

---

### Module 2: Artificial Neural Networks (Practicals 2.1 – 2.10)
| # | Practical Script | Core Topic / Aim | Output Plot |
|:---:|---|---|---|
| **2.1** | [`ex01_artificial_neuron.py`](Module_2_Neural_Networks/ex01_artificial_neuron.py) | Mathematical model of an Artificial Neuron (Inputs, Weights, Activation) | `mod2_ex01_artificial_neuron.png` |
| **2.2** | [`ex02_weighted_sum_and_bias.py`](Module_2_Neural_Networks/ex02_weighted_sum_and_bias.py) | Weighted sum, bias shifting effect, and step response | `mod2_ex02_weighted_sum_and_bias.png` |
| **2.3** | [`ex03_activation_functions.py`](Module_2_Neural_Networks/ex03_activation_functions.py) | Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax, ELU & Derivatives | `mod2_ex03_activation_functions.png` |
| **2.4** | [`ex04_single_layer_network.py`](Module_2_Neural_Networks/ex04_single_layer_network.py) | Single-Layer Perceptron for linearly separable logic gates (AND, OR) | `mod2_ex04_single_layer_network.png` |
| **2.5** | [`ex05_mlp_keras.py`](Module_2_Neural_Networks/ex05_mlp_keras.py) | Multi-Layer Perceptron (MLP) architecture using Keras/PyTorch for XOR | `mod2_ex05_mlp_keras.png` |
| **2.6** | [`ex06_gradient_descent.py`](Module_2_Neural_Networks/ex06_gradient_descent.py) | Gradient Descent optimization visualization & learning rate trajectory | `mod2_ex06_gradient_descent.png` |
| **2.7** | [`ex07_backpropagation.py`](Module_2_Neural_Networks/ex07_backpropagation.py) | Mathematical Backpropagation algorithm with MSE loss reduction curve | `mod2_ex07_backpropagation.png` |
| **2.8** | [`ex08_train_model_keras.py`](Module_2_Neural_Networks/ex08_train_model_keras.py) | Model training workflow (Compile, Fit, Loss/Accuracy convergence) | `mod2_ex08_train_model_keras.png` |
| **2.9** | [`ex09_evaluation_metrics.py`](Module_2_Neural_Networks/ex09_evaluation_metrics.py) | Classification evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix) | `mod2_ex09_evaluation_metrics.png` |
| **2.10** | [`ex10_complete_ann_classification.py`](Module_2_Neural_Networks/ex10_complete_ann_classification.py) | Complete End-to-End ANN system for California Housing price classification | `mod2_ex10_complete_ann_system.png` |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Create and activate Python 3.11/3.12 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all required scientific and ML dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Suite
Verify all 20 scripts, 20 plots, module notebooks, master notebook, and lab manuals in one command:
```bash
python verify_assignment_2.py
```

### 3. Run Individual Practical Exercises
Any script can be executed independently from the repository root:
```bash
# Example: Run Practical 1.10 (Fuzzy Controller)
python Module_1_Fuzzy_Logic/ex10_fuzzy_logic_controller.py

# Example: Run Practical 2.10 (Complete ANN System)
python Module_2_Neural_Networks/ex10_complete_ann_classification.py
```

### 4. Regenerate Lab Manual Document (.docx & .md)
```bash
python generate_lab_manual.py
```

### 5. Re-execute Jupyter Notebooks
```bash
# Rebuild and execute master consolidated notebook
python build_jupyter_notebook.py

# Rebuild both individual module notebooks
python build_all_module_notebooks.py
```
