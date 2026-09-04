"""
Practical 2.8: Training a Neural Network Model on a Dataset Using Keras
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To train a simple neural network model on a dataset using Keras/TensorFlow.

Theory:
Supervised training of neural networks on real datasets involves:
1. Data Preprocessing: Feature normalization (Z-score scaling) and categorical one-hot encoding.
2. Architecture Design: Selecting appropriate layer depth, width, and activation functions.
3. Loss Function & Optimizer: Cross-Entropy loss for classification with adaptive gradient methods (Adam).
4. Validation Monitoring: Tracking training loss vs. validation loss over epochs to detect overfitting.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras import layers

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_practical_2_8():
    print("=" * 60)
    print("PRACTICAL 2.8: Keras Model Training on Iris Dataset")
    print("=" * 60)

    # 1. Load Dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Dataset: Iris ({len(X)} samples, {X.shape[1]} features, {len(target_names)} classes)")

    # 2. Train-Test Split & Feature Normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Architecture
    model = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(16, activation='relu', name='dense_hidden1'),
        layers.Dense(8, activation='relu', name='dense_hidden2'),
        layers.Dense(3, activation='softmax', name='output_softmax')
    ], name="Iris_Classification_MLP")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.03),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Model Architecture ---")
    model.summary()

    # 4. Train Model
    epochs = 80
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=16,
        verbose=0
    )

    # 5. Evaluate on Test Set
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nFinal Test Set Loss:     {test_loss:.4f}")
    print(f"Final Test Set Accuracy: {test_acc * 100.0:.2f}%\n")

    # 6. Visualization: Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves
    ax1.plot(range(1, epochs + 1), history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(range(1, epochs + 1), history.history['val_loss'], 'r--', linewidth=2, label='Validation Loss')
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Accuracy Curves
    ax2.plot(range(1, epochs + 1), history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(range(1, epochs + 1), history.history['val_accuracy'], 'r--', linewidth=2, label='Validation Accuracy')
    ax2.axhline(test_acc, color='green', linestyle=':', label=f'Test Accuracy ({test_acc*100:.1f}%)')
    ax2.set_title("Classification Accuracy vs. Epochs", fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex08_train_model_keras.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.8 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_8()
