"""
Linear regression, solved three ways.

We fit  y = X b + noise  by least squares, i.e. we find the b minimising
||X b - y||^2, using three different methods:

    1. Normal equations   -- solve (X'X) b = X'y
    2. QR decomposition   -- the numerically stable direct method
    3. Gradient descent   -- the iterative method used in machine learning

All three solve the same problem. On easy data they agree to 12+ decimal
places. On badly conditioned data they do NOT, and the difference is large
and predictable. That is what this project is about.

Everything here is written from scratch. NumPy is used for array arithmetic
and for the QR factorisation itself; scikit-learn appears only once, as an
independent check that our answers are right.
"""

import numpy as np


# ==========================================================================
# METHOD 1: normal equations
# ==========================================================================

def fit_normal_equations(X, y):
    """Solve (X'X) b = X'y.

    Setting the gradient of ||Xb - y||^2 to zero gives X'X b = X'y, so
    b = (X'X)^{-1} X'y. This is the formula in every textbook, and it works
    fine most of the time.

    The catch: forming X'X SQUARES the condition number of X. If X is
    already somewhat ill-conditioned, X'X can be disastrous. Experiment 2
    measures exactly how much accuracy this costs.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)      # solve, never invert explicitly


# ==========================================================================
# METHOD 2: QR decomposition
# ==========================================================================

def fit_qr(X, y):
    """Solve via X = QR, where Q has orthonormal columns and R is upper
    triangular.

    Then  ||Xb - y||^2 = ||QRb - y||^2 = ||Rb - Q'y||^2   (Q preserves
    lengths), so we just solve the triangular system R b = Q'y by back
    substitution.

    Crucially we never form X'X, so the condition number is not squared.
    This is what professional least-squares solvers actually do.
    """
    Q, R = np.linalg.qr(X)
    return back_substitution(R, Q.T @ y)


def back_substitution(R, z):
    """Solve R b = z for upper-triangular R, by hand.

    Work from the bottom row upward: the last row involves only b[-1], so
    solve for it, substitute into the row above, and repeat.
    """
    n = R.shape[1]
    b = np.zeros(n)
    for i in range(n - 1, -1, -1):
        b[i] = (z[i] - R[i, i + 1:] @ b[i + 1:]) / R[i, i]
    return b


# ==========================================================================
# METHOD 3: gradient descent
# ==========================================================================

def fit_gradient_descent(X, y, lr=0.01, n_iter=2000):
    """Minimise  L(b) = (1/n) ||Xb - y||^2  by repeated small steps downhill.

    The gradient is

        dL/db = (2/n) X' (X b - y)

    and we update  b <- b - lr * gradient.

    Returns the fitted coefficients and the loss at every iteration, so we
    can watch it converge.

    This is the method that scales to data too large to factorise, and it is
    the ancestor of the optimiser inside every neural network.
    """
    n, p = X.shape
    b = np.zeros(p)
    history = []
    for _ in range(n_iter):
        residual = X @ b - y
        history.append(float(np.mean(residual ** 2)))
        gradient = (2.0 / n) * (X.T @ residual)
        b = b - lr * gradient
    return b, np.array(history)


# ==========================================================================
# helpers
# ==========================================================================

def r_squared(X, y, b):
    """Fraction of variance explained:  1 - SS_res / SS_tot."""
    resid = y - X @ b
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def make_data(n=200, p=3, noise=0.5, seed=0):
    """Well-conditioned synthetic data with known true coefficients."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
    b_true = rng.normal(size=p + 1) * 3
    y = X @ b_true + rng.normal(0, noise, size=n)
    return X, y, b_true


def make_ill_conditioned(n=100, p=8, seed=0):
    """A Vandermonde matrix: columns are 1, t, t^2, ..., t^(p-1).

    Polynomial fitting in this basis is the classic ill-conditioned problem,
    because t^7 and t^8 look almost identical on [0, 1] -- the columns are
    nearly linearly dependent, so the condition number is enormous.

    Note there is NO noise here: y = X b_true exactly. That is deliberate.
    With noise, the error from amplifying that noise swamps the error from
    floating-point arithmetic, and both methods look equally bad. Removing
    the noise isolates the thing we actually want to measure: how much
    accuracy each ALGORITHM loses on its own.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = np.vander(t, p, increasing=True)
    b_true = rng.normal(size=p)
    y = X @ b_true
    return X, y, b_true
