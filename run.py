"""
Run the experiments.   python run.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regression import (
    fit_normal_equations, fit_qr, fit_gradient_descent,
    r_squared, make_data, make_ill_conditioned,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "figure.autolayout": True})


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ==========================================================================
# 1. do the three methods agree, and are they right?
# ==========================================================================

line("EXPERIMENT 1  Three methods, well-conditioned data")

X, y, b_true = make_data(n=200, p=3, noise=0.5, seed=0)

b_ne = fit_normal_equations(X, y)
b_qr = fit_qr(X, y)
b_gd, history = fit_gradient_descent(X, y, lr=0.05, n_iter=5000)

print(f"\ncondition number of X: {np.linalg.cond(X):.2f}  (small = easy)\n")
print(f"{'coefficient':<14}{'true':>10}{'normal eq':>12}{'QR':>12}{'grad desc':>12}")
for i in range(len(b_true)):
    print(f"b[{i}]{'':<10}{b_true[i]:>10.4f}{b_ne[i]:>12.4f}"
          f"{b_qr[i]:>12.4f}{b_gd[i]:>12.4f}")

print(f"\nR-squared: {r_squared(X, y, b_qr):.4f}")
print(f"max difference between normal equations and QR: "
      f"{np.max(np.abs(b_ne - b_qr)):.2e}")
print(f"max difference between gradient descent and QR: "
      f"{np.max(np.abs(b_gd - b_qr)):.2e}")

# independent check
try:
    from sklearn.linear_model import LinearRegression
    sk = LinearRegression(fit_intercept=False).fit(X, y)
    print(f"max difference from scikit-learn:               "
          f"{np.max(np.abs(sk.coef_ - b_qr)):.2e}")
except ImportError:
    print("(scikit-learn not installed, skipping cross-check)")

print("\nOn easy data all three agree to ~10 decimal places. Good -- they")
print("are solving the same problem, and our implementations are correct.")


# ==========================================================================
# 2. ill-conditioned data: now they disagree
# ==========================================================================

line("EXPERIMENT 2  Ill-conditioned data: the methods come apart")

print(f"\n{'degree':>8}{'cond(X)':>12}{'cond(X\'X)':>13}"
      f"{'normal eq err':>16}{'QR err':>14}")

degrees, err_ne, err_qr, conds = [], [], [], []
for p in range(4, 13):
    Xi, yi, b_t = make_ill_conditioned(n=100, p=p, seed=1)
    c = np.linalg.cond(Xi)
    c2 = np.linalg.cond(Xi.T @ Xi)

    try:
        e_ne = float(np.max(np.abs(fit_normal_equations(Xi, yi) - b_t)))
    except np.linalg.LinAlgError:
        e_ne = float("inf")
    e_qr = float(np.max(np.abs(fit_qr(Xi, yi) - b_t)))

    print(f"{p:>8}{c:>12.2e}{c2:>13.2e}{e_ne:>16.2e}{e_qr:>14.2e}")
    degrees.append(p); err_ne.append(e_ne); err_qr.append(e_qr); conds.append(c)

print("\ncond(X'X) = cond(X)^2. Forming X'X squares the conditioning, and")
print("the normal equations lose roughly twice as many digits as QR does.")
print("This is the entire reason real solvers use QR or SVD instead of the")
print("textbook formula.")

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.semilogy(degrees, err_ne, "o-", label="Normal equations")
ax.semilogy(degrees, err_qr, "s-", label="QR decomposition")
ax.set_xlabel("polynomial degree (higher = worse conditioned)")
ax.set_ylabel("max coefficient error")
ax.set_title("Normal equations lose accuracy faster than QR")
ax.legend(fontsize=8)
fig.savefig("fig_conditioning.png", bbox_inches="tight")
plt.close(fig)


# ==========================================================================
# 3. gradient descent: the learning rate
# ==========================================================================

line("EXPERIMENT 3  Gradient descent and the learning rate")

X, y, _ = make_data(n=200, p=3, noise=0.5, seed=0)
b_exact = fit_qr(X, y)

# largest stable learning rate is 1 / L, where L is the largest eigenvalue
# of the Hessian (2/n) X'X
L = 2.0 / X.shape[0] * np.linalg.eigvalsh(X.T @ X).max()
print(f"\nLargest eigenvalue of the Hessian: L = {L:.4f}")
print(f"Theory: gradient descent converges for learning rate < 2/L = {2/L:.4f}\n")

print(f"{'learning rate':>14}{'final error':>16}{'behaviour':>14}")
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for lr in [0.001, 0.01, 0.05, 0.2, 0.5, 1.05 * 2 / L]:
    with np.errstate(over="ignore", invalid="ignore"):
        b, hist = fit_gradient_descent(X, y, lr=lr, n_iter=3000)
    err = float(np.max(np.abs(b - b_exact)))
    if not np.isfinite(err) or err > 1e3:
        ok = "DIVERGED"
    elif err < 1e-3:
        ok = "converged"
    else:
        ok = "too slow"
    print(f"{lr:>14.4f}{err:>16.2e}{ok:>14}")
    if np.all(np.isfinite(hist)):
        ax.semilogy(hist[:600], lw=1.1, label=f"lr = {lr:.3f}")

ax.set_xlabel("iteration")
ax.set_ylabel("mean squared error")
ax.set_title("Too small = slow. Too large = divergence.")
ax.legend(fontsize=7)
fig.savefig("fig_gradient_descent.png", bbox_inches="tight")
plt.close(fig)

print("\nThe learning rate is not a magic number to be tuned by trial and")
print("error -- it has a computable upper bound set by the curvature of the")
print("loss surface.")

print("\nSaved: fig_conditioning.png, fig_gradient_descent.png")
