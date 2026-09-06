# Linear Least Squares, Solved Three Ways

Fitting `y = Xb` by least squares using three different methods, and measuring
where they agree and where they don't.

All three solve the same optimisation problem. On easy data they agree to 15
decimal places. On badly conditioned data one of them loses seven orders of
magnitude more accuracy than the other, in a way that is predicted exactly by
the condition number.

The methods are implemented from scratch (including back substitution).
scikit-learn is used once, as an independent check that the answers are right.

## The three methods

| Method | Idea | Cost |
|---|---|---|
| Normal equations | Set the gradient to zero, solve `X'X b = X'y` | Cheapest |
| QR decomposition | Factor `X = QR`, then solve `Rb = Q'y` by back substitution | ~2× more |
| Gradient descent | Step downhill: `b ← b − lr · ∇L(b)` | Iterative |

## Result 1: on easy data, all three agree

Condition number of X: 1.14.

| | true | normal eq. | QR | gradient descent |
|---|---|---|---|---|
| b₀ | −3.4705 | −3.4843 | −3.4843 | −3.4843 |
| b₁ | −4.0946 | −4.0184 | −4.0184 | −4.0184 |
| b₂ | −0.6937 | −0.6823 | −0.6823 | −0.6823 |
| b₃ | 6.8325 | 6.7947 | 6.7947 | 6.7947 |

R² = 0.9964. Maximum disagreement between any two methods: **3.6×10⁻¹⁵** —
floating-point noise. They also match scikit-learn to 4.4×10⁻¹⁵, which confirms
the implementations are correct.

(The fitted values differ from the true ones by ~0.01 because the data has
observation noise. That is statistical error, not numerical error — a different
thing entirely, and worth not confusing.)

## Result 2: on hard data, they come apart

Fitting a polynomial of increasing degree in the basis `1, t, t², ...` on
[0, 1]. High powers of t look nearly identical there, so the columns are almost
linearly dependent and the condition number explodes.

There is **no observation noise** in this experiment — `y = Xb` exactly. That is
deliberate: with noise, the error from amplifying the noise swamps the error
from the arithmetic, and both methods look equally bad. Removing it isolates
what we actually want to measure.

| degree | cond(X) | cond(X'X) | normal eq. error | QR error |
|---|---|---|---|---|
| 4 | 1.2e+02 | 1.5e+04 | 7.9e−14 | 8.3e−16 |
| 6 | 3.7e+03 | 1.4e+07 | 8.7e−11 | 1.6e−13 |
| 8 | 1.2e+05 | 1.4e+10 | 1.1e−07 | 6.4e−13 |
| 10 | 3.7e+06 | 1.4e+13 | 1.1e−04 | 1.7e−10 |
| 12 | 1.2e+08 | 1.3e+16 | **5.6e−02** | **2.7e−09** |

At degree 12 the normal equations have lost essentially all accuracy while QR is
still good to nine digits.

**Why:** `cond(X'X) = cond(X)²`. Forming `X'X` squares the conditioning, and the
achievable accuracy is roughly `machine epsilon × condition number`. So the
normal equations lose about twice as many digits as a method that never forms
`X'X`. This is why production least-squares solvers use QR or SVD rather than
the formula from the textbook.

## Result 3: the learning rate has a computable limit

For `L(b) = (1/n)‖Xb − y‖²`, the Hessian is `(2/n)X'X`. Gradient descent
converges when the learning rate satisfies `lr < 2/L`, where `L` is the largest
eigenvalue of that Hessian.

Measured: L = 2.2069, so the predicted threshold is **2/L = 0.9063**.

| learning rate | final error | behaviour |
|---|---|---|
| 0.001 | 1.6e−02 | too slow |
| 0.010 | 2.5e−14 | converged |
| 0.050 | 3.6e−15 | converged |
| 0.200 | 8.9e−16 | converged |
| 0.500 | 1.8e−15 | converged |
| 0.952 | 4.9e+124 | **diverged** |

The last row is just above the threshold, and it blows up. The learning rate is
not a hyperparameter to be found by trial and error here — it is bounded by the
curvature of the loss surface, and the bound is exact.

## Files

```
regression.py   the three methods, plus back substitution, written from scratch
run.py          the three experiments
fig_conditioning.png
fig_gradient_descent.png
```

## Running

```bash
pip install numpy matplotlib scikit-learn
python run.py
```

Runs in a few seconds.
