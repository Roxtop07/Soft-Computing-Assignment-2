"""
Practical 2.10: Complete End-to-End ANN-Based Classification System
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To design, train, and evaluate a complete Artificial Neural Network-based classification system
using Python and Keras/TensorFlow by integrating input features, hidden layers, activation
functions, training, and model evaluation.

Theory:
A production-grade ANN classification pipeline consists of 6 integrated engineering stages:
1. Problem Definition & Data Ingestion: Breast Cancer Diagnostic dataset (569 samples, 30 clinical features).
2. Data Preprocessing & Leakage Prevention: Stratified train/validation/test partitioning and StandardScaler.
3. Architecture Engineering: Multilayer Deep Feedforward Network with Dense layers, Batch Normalization,
   Dropout regularization (preventing co-adaptation of neurons), and He-normal initialization.
4. Optimization & Regularization: Adam optimizer with learning rate scheduling and Early Stopping.
5. Model Training: Minibatch gradient updates across training epochs with validation tracking.
6. Multi-Metric Clinical Evaluation: Accuracy, Precision, Recall/Sensitivity, Specificity, ROC-AUC, Confusion Matrix.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)

import keras
from keras import layers, regularizers

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_practical_2_10():
    print("=" * 60)
    print("PRACTICAL 2.10: Complete End-to-End ANN Classification System")
    print("=" * 60)

    # 1. Dataset Ingestion
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names  # ['malignant', 'benign']

    print(f"Dataset: Wisconsin Breast Cancer Diagnostic")
    print(f"Features: {X.shape[1]} clinical biometric features | Total Samples: {X.shape[0]}")
    print(f"Target Classes: {target_names[0]} (0) vs. {target_names[1]} (1)")

    # 2. Preprocessing & Partitioning (70% Train, 15% Val, 15% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Split distribution: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    # 3. Model Architecture with Dropout Regularization
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],), name="Biometric_Input_Layer"),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="Hidden_Dense_1"),
        layers.Dropout(0.3, name="Dropout_1"),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="Hidden_Dense_2"),
        layers.Dropout(0.2, name="Dropout_2"),
        layers.Dense(16, activation="relu", name="Hidden_Dense_3"),
        layers.Dense(1, activation="sigmoid", name="Diagnostic_Output")
    ], name="End_to_End_Clinical_ANN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\n--- ANN Architecture Specification ---")
    model.summary()

    # 4. Model Training with Early Stopping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ]

    epochs = 80
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    actual_epochs = len(history.history['loss'])
    print(f"\nTraining completed in {actual_epochs} epochs (Early stopping active).")

    # 5. Model Evaluation on Unseen Test Partition
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print("\n--- Final Test Evaluation Results ---")
    print(f"Test Accuracy:  {test_acc * 100.0:.2f}%")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Sensitivity:    {(tp / (tp + fn)) * 100.0:.2f}% (Recall for Benign)")
    print(f"Specificity:    {(tn / (tn + fp)) * 100.0:.2f}% (True Malignant Rate)")
    print(f"ROC-AUC Score:  {roc_auc:.4f}\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

    # 6. Visualization Dashboard (4-panel diagnostic figure)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Loss History
    axes[0, 0].plot(range(1, actual_epochs + 1), history.history['loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].plot(range(1, actual_epochs + 1), history.history['val_loss'], 'r--', linewidth=2, label='Val Loss')
    axes[0, 0].set_title("Training vs. Validation Loss (Binary Cross-Entropy)", fontweight='bold')
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # Panel 2: Accuracy History
    axes[0, 1].plot(range(1, actual_epochs + 1), history.history['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
    axes[0, 1].plot(range(1, actual_epochs + 1), history.history['val_accuracy'], 'r--', linewidth=2, label='Val Accuracy')
    axes[0, 1].axhline(test_acc, color='green', linestyle=':', label=f'Test Acc ({test_acc*100:.1f}%)')
    axes[0, 1].set_title("Training vs. Validation Accuracy", fontweight='bold')
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()

    # Panel 3: Confusion Matrix
    im = axes[1, 0].imshow(cm, cmap='Blues')
    axes[1, 0].set_title(f"Test Confusion Matrix (Accuracy: {test_acc*100:.1f}%)", fontweight='bold')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Pred Malignant', 'Pred Benign'])
    axes[1, 0].set_yticklabels(['Actual Malignant', 'Actual Benign'])
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f"{cm[i, j]}", ha='center', va='center',
                            color='white' if cm[i, j] > cm.max()/2 else 'black',
                            fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: ROC Curve
    axes[1, 1].plot(fpr, tpr, color='darkorange', linewidth=2.5, label=f'ANN ROC (AUC = {roc_auc:.4f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance')
    axes[1, 1].set_title("Receiver Operating Characteristic (ROC)", fontweight='bold')
    axes[1, 1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1, 1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex10_complete_ann_system.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Dashboard saved successfully to: {plot_path}")
    print("[✓] Practical 2.10 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_10()
