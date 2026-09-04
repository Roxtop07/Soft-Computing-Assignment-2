"""
Practical 2.5: Multilayer Perceptron (MLP) with Input, Hidden, and Output Layers
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To create a Multilayer Perceptron (MLP) with input, hidden, and output layers using Keras/TensorFlow.

Theory:
A Multilayer Perceptron (MLP) overcomes the linear separability limitations of single-layer perceptrons
(Minsky & Papert, 1969) by inserting one or more hidden layers with non-linear activation functions.
Architecture:
- Input Layer: Accepts d features (e.g. x1, x2).
- Hidden Layer(s): Applies affine transformation followed by non-linear activation:
      h = ReLU(W1 * x + b1)
- Output Layer: Maps hidden representations to target predictions:
      y_hat = Sigmoid(W2 * h + b2)
Universal Approximation Theorem guarantees that an MLP with at least one non-linear hidden layer
can approximate any continuous function.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import keras
    from keras import layers
    HAS_KERAS = True
except Exception as e:
    HAS_KERAS = False

def run_practical_2_5():
    print("=" * 60)
    print("PRACTICAL 2.5: Multilayer Perceptron (MLP) Architecture")
    print("=" * 60)
    print(f"Keras framework available: {HAS_KERAS}")

    # Dataset: Non-linear XOR Problem (Cannot be solved by single-layer perceptron)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    if HAS_KERAS:
        # Define MLP Model using Keras Sequential API
        model = keras.Sequential([
            layers.Input(shape=(2,), name="Input_Layer"),
            layers.Dense(units=8, activation="relu", name="Hidden_Layer_1"),
            layers.Dense(units=4, activation="relu", name="Hidden_Layer_2"),
            layers.Dense(units=1, activation="sigmoid", name="Output_Layer")
        ], name="Multilayer_Perceptron_XOR")

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.08),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("\n--- Keras MLP Model Summary ---")
        model.summary()

        # Train model
        history = model.fit(X, y, epochs=250, verbose=0)
        predictions = model.predict(X, verbose=0)
    else:
        # Pure NumPy MLP for XOR fallback
        np.random.seed(42)
        W1 = np.random.randn(2, 4)
        b1 = np.zeros((1, 4))
        W2 = np.random.randn(4, 1)
        b2 = np.zeros((1, 1))

        def sig(z): return 1 / (1 + np.exp(-z))
        for _ in range(5000):
            z1 = X @ W1 + b1
            a1 = np.maximum(0, z1) # relu
            z2 = a1 @ W2 + b2
            a2 = sig(z2)
            # Backprop
            d2 = a2 - y
            dW2 = a1.T @ d2
            db2 = np.sum(d2, axis=0, keepdims=True)
            d1 = (d2 @ W2.T) * (z1 > 0)
            dW1 = X.T @ d1
            db1 = np.sum(d1, axis=0, keepdims=True)
            W1 -= 0.1 * dW1
            b1 -= 0.1 * db1
            W2 -= 0.1 * dW2
            b2 -= 0.1 * db2
        predictions = sig(np.maximum(0, X @ W1 + b1) @ W2 + b2)

    print("\nNon-linear XOR Truth Table & MLP Predictions:")
    print(f"{'x1':^5} | {'x2':^5} | {'y_true':^8} | {'y_pred (prob)':^15} | {'Rounded':^8}")
    print("-" * 52)
    for i in range(4):
        p = float(predictions[i][0])
        print(f"{int(X[i, 0]):^5} | {int(X[i, 1]):^5} | {int(y[i, 0]):^8} | {p:^15.4f} | {round(p):^8}")

    # Plot Decision Surface
    x_grid = np.linspace(-0.5, 1.5, 200)
    y_grid = np.linspace(-0.5, 1.5, 200)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[XX.ravel(), YY.ravel()]

    if HAS_KERAS:
        ZZ = model.predict(grid_points, verbose=0).reshape(XX.shape)
    else:
        ZZ = sig(np.maximum(0, grid_points @ W1 + b1) @ W2 + b2).reshape(XX.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(XX, YY, ZZ, levels=20, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(contour, label='Predicted Output Probability')

    # Overlay ground truth XOR points
    for i in range(4):
        color = 'green' if y[i, 0] == 1 else 'red'
        marker = 'o' if y[i, 0] == 1 else 's'
        plt.scatter(X[i, 0], X[i, 1], c=color, s=150, marker=marker, edgecolors='black',
                    linewidths=2, label=f"True Class {int(y[i, 0])}" if i < 2 else "")

    plt.title("MLP Non-Linear Decision Surface for XOR Problem", fontsize=12, fontweight='bold')
    plt.xlabel("Input Feature x1", fontsize=11)
    plt.ylabel("Input Feature x2", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex05_mlp_keras.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.5 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_5()
