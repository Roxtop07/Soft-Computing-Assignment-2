"""
Practical 2.6: Gradient Descent Optimization (Batch, Stochastic, and Mini-Batch)
Course: Soft Computing (EL1) - STDA2102 | Module 2: Artificial Neural Networks

Aim:
To apply Gradient Descent for updating the weights of a simple neural network during training.

Theory:
Gradient Descent iteratively updates model parameters θ in the direction of steepest descent
of the loss function J(θ):
    θ_{t+1} = θ_t - η * ∇_θ J(θ_t)

Three Primary Variants:
1. Batch Gradient Descent (BGD): Uses the entire dataset of N samples per step.
   Smooth, stable convergence, but computationally expensive for large datasets.
2. Stochastic Gradient Descent (SGD): Uses a single training instance (N=1) per update step.
   Fast and can escape local minima, but exhibits high variance/oscillation.
3. Mini-Batch Gradient Descent (MBGD): Uses mini-batches (e.g., size B=16 or 32).
   Achieves balance between computational efficiency of vectorized ops and regularized convergence.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_regression_data(n_samples=200):
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    # Ground truth: y = 4 + 3*X + noise
    y = 4 + 3 * X + np.random.randn(n_samples, 1) * 0.5
    return X, y

def compute_loss(X_b, y, theta):
    m = len(y)
    preds = X_b.dot(theta)
    return (1.0 / (2 * m)) * np.sum((preds - y) ** 2)

def run_practical_2_6():
    print("=" * 60)
    print("PRACTICAL 2.6: Gradient Descent Algorithms Comparison")
    print("=" * 60)

    X, y = generate_regression_data(n_samples=200)
    X_b = np.c_[np.ones((len(X), 1)), X]  # add bias feature x0 = 1
    m = len(y)

    lr = 0.05
    n_epochs = 50

    # 1. Batch Gradient Descent
    theta_bgd = np.array([[0.0], [0.0]])
    bgd_losses = []
    bgd_thetas = [theta_bgd.copy()]
    for _ in range(n_epochs):
        gradients = (1.0 / m) * X_b.T.dot(X_b.dot(theta_bgd) - y)
        theta_bgd -= lr * gradients
        bgd_thetas.append(theta_bgd.copy())
        bgd_losses.append(compute_loss(X_b, y, theta_bgd))

    # 2. Stochastic Gradient Descent (SGD)
    theta_sgd = np.array([[0.0], [0.0]])
    sgd_losses = []
    sgd_thetas = [theta_sgd.copy()]
    for epoch in range(n_epochs):
        indices = np.random.permutation(m)
        for i in indices:
            xi = X_b[i:i+1]
            yi = y[i:i+1]
            gradients = xi.T.dot(xi.dot(theta_sgd) - yi)
            theta_sgd -= (lr * 0.5) * gradients
        sgd_thetas.append(theta_sgd.copy())
        sgd_losses.append(compute_loss(X_b, y, theta_sgd))

    # 3. Mini-batch Gradient Descent
    theta_mb = np.array([[0.0], [0.0]])
    mb_losses = []
    batch_size = 20
    mb_thetas = [theta_mb.copy()]
    for epoch in range(n_epochs):
        shuffled = np.random.permutation(m)
        X_shuff = X_b[shuffled]
        y_shuff = y[shuffled]
        for i in range(0, m, batch_size):
            xb_i = X_shuff[i:i+batch_size]
            yb_i = y_shuff[i:i+batch_size]
            gradients = (1.0 / len(yb_i)) * xb_i.T.dot(xb_i.dot(theta_mb) - yb_i)
            theta_mb -= lr * gradients
        mb_thetas.append(theta_mb.copy())
        mb_losses.append(compute_loss(X_b, y, theta_mb))

    print(f"Ground Truth Model: y = 4.0 + 3.0 * x")
    print(f"Batch GD Estimated:       bias={theta_bgd[0, 0]:.3f}, weight={theta_bgd[1, 0]:.3f} | Final Loss={bgd_losses[-1]:.4f}")
    print(f"Stochastic GD Estimated:  bias={theta_sgd[0, 0]:.3f}, weight={theta_sgd[1, 0]:.3f} | Final Loss={sgd_losses[-1]:.4f}")
    print(f"Mini-Batch GD Estimated:  bias={theta_mb[0, 0]:.3f}, weight={theta_mb[1, 0]:.3f} | Final Loss={mb_losses[-1]:.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Loss curves
    epochs_range = range(1, n_epochs + 1)
    ax1.plot(epochs_range, bgd_losses, 'b-', linewidth=2.2, label=f'Batch GD (Loss={bgd_losses[-1]:.3f})')
    ax1.plot(epochs_range, sgd_losses, 'r--', linewidth=1.8, label=f'Stochastic GD (Loss={sgd_losses[-1]:.3f})')
    ax1.plot(epochs_range, mb_losses, 'g-.', linewidth=2.0, label=f'Mini-Batch GD (Loss={mb_losses[-1]:.3f})')
    ax1.set_title("Loss Convergence (MSE) vs. Epochs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Mean Squared Error", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot 2: Trajectory in Parameter Space (b, w)
    bgd_t = np.array(bgd_thetas)[:, :, 0]
    sgd_t = np.array(sgd_thetas)[:, :, 0]
    mb_t = np.array(mb_thetas)[:, :, 0]

    ax2.plot(bgd_t[:, 0], bgd_t[:, 1], 'bo-', label='Batch GD Trajectory', linewidth=1.5, markersize=3)
    ax2.plot(sgd_t[:, 0], sgd_t[:, 1], 'r.-', label='Stochastic GD Trajectory', alpha=0.7, linewidth=1.2)
    ax2.plot(mb_t[:, 0], mb_t[:, 1], 'g^-', label='Mini-Batch GD Trajectory', linewidth=1.5, markersize=3)
    ax2.scatter(4.0, 3.0, color='gold', s=180, marker='*', edgecolors='black', label='Optimal (b=4, w=3)', zorder=5)
    ax2.set_title("Optimization Path in Parameter Space (Intercept vs Slope)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Parameter θ0 (Bias)", fontsize=11)
    ax2.set_ylabel("Parameter θ1 (Weight)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mod2_ex06_gradient_descent.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[✓] Plot saved successfully to: {plot_path}")
    print("[✓] Practical 2.6 completed successfully.\n")

if __name__ == "__main__":
    run_practical_2_6()
