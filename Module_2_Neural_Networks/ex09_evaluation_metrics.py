"""
Practical 2.9: Evaluation Metrics for Neural Networks
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To evaluate the performance of a trained neural network using suitable evaluation metrics in Python.

Theory:
For binary and multi-class neural classifiers, performance is evaluated using:
1. Confusion Matrix: Contingency table of actual vs. predicted labels (TP, FP, TN, FN).
2. Accuracy: (TP + TN) / (TP + TN + FP + FN)
3. Precision (Positive Predictive Value): TP / (TP + FP)
4. Recall (Sensitivity / Hit Rate): TP / (TP + FN)
5. F1-Score: Harmonic mean of precision and recall: 2 * (Precision * Recall) / (Precision + Recall)
6. ROC Curve (Receiver Operating Characteristic): Plots True Positive Rate (Sensitivity)
   against False Positive Rate (1 - Specificity) across decision thresholds.
7. AUC (Area Under the ROC Curve): Overall separability metric (1.0 = perfect, 0.5 = random guess).
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)

import keras
from keras import layers

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_practical_2_9():
    print("=" * 60)
    print("PRACTICAL 2.9: Neural Network Performance Evaluation Metrics")
    print("=" * 60)

    # 1. Generate Realistic Imbalanced Dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=7,
                               weights=[0.7, 0.3], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2. Train Neural Classifier
    model = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)

    # 3. Compute Predictions
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # 4. Compute Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)

    print("\n--- Quantitative Evaluation Summary ---")
    print(f"True Positives (TP):  {tp:4d} | False Positives (FP): {fp:4d}")
    print(f"True Negatives (TN):  {tn:4d} | False Negatives (FN): {fn:4d}")
    print(f"Accuracy:   {accuracy * 100.0:.2f}%")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0 (Negative)', 'Class 1 (Positive)']))

    # 5. Visualization: Confusion Matrix & ROC Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Confusion Matrix Heatmap
    im = ax1.imshow(cm, cmap='Blues', interpolation='nearest')
    ax1.set_title("Confusion Matrix Heatmap", fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Pred Negative (0)', 'Pred Positive (1)'])
    ax1.set_yticklabels(['Actual Negative (0)', 'Actual Positive (1)'])
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Annotate values
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax1.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                     color=color, fontsize=14, fontweight='bold')

    # ROC Curve
    ax2.plot(fpr, tpr, color='darkorange', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.500)')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax2.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex09_evaluation_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.9 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_9()
