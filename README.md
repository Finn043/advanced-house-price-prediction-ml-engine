# NumPy Linear Regression — Interview Cheat Sheet

A from-scratch linear regression engine for Melbourne house prices. The implementation uses NumPy for the model and optimization math, with explicit MSE, MAE, and Huber losses plus Batch Gradient Descent and Stochastic Gradient Descent (SGD).

## Linear Regression

For `m` examples and `n` features:

\[
\hat{y} = Xw + b
\]

- `X ∈ ℝ^{m×n}`: feature matrix; `w ∈ ℝ^n`: coefficients; `b ∈ ℝ`: bias.
- Each prediction is `ŷᵢ = xᵢᵀw + b`.
- The trainer standardizes numeric features and target, and one-hot encodes `Type` and `Region`.

## Loss Functions

Let residual `eᵢ = ŷᵢ − yᵢ`, with `m` observations.

| Loss | Objective `J` | Gradient w.r.t. prediction `ŷᵢ` | Interview takeaway |
| --- | --- | --- | --- |
| **MSE** | `\(\frac{1}{m}\sum_i e_i^2\)` | `\(\frac{2e_i}{m}\)` | Smooth gradient; squaring makes large errors and outliers disproportionately influential. |
| **MAE** | `\(\frac{1}{m}\sum_i |e_i|\)` | `\(\frac{\operatorname{sign}(e_i)}{m}\)` | More robust to outliers; not differentiable at zero, so this implementation uses the subgradient `sign(0) = 0`. |
| **Huber** | `\(\frac{1}{m}\sum_i L_\delta(e_i)\)` | `\(\frac{\operatorname{clip}(e_i,-\delta,\delta)}{m}\)` | Quadratic for small residuals and linear for large ones; balances smoothness with outlier robustness. |

\[
L_\delta(e) =
\begin{cases}
\frac{1}{2}e^2 & |e| \leq \delta \\
\delta(|e| - \frac{1}{2}\delta) & |e| > \delta
\end{cases}
\]

The trainer uses `δ = 1` in standardized target units. Loss and prediction gradients are implemented in [`model/loss_functions.py`](model/loss_functions.py).

## Gradient Descent

By the chain rule, for prediction gradient `g = ∂J/∂ŷ`:

\[
\nabla_w J = X^Tg, \qquad \frac{\partial J}{\partial b} = \sum_i g_i
\]

Update parameters in the direction opposite the gradient:

\[
w \leftarrow w - \alpha\nabla_w J, \qquad b \leftarrow b - \alpha\frac{\partial J}{\partial b}
\]

`α` is the learning rate. Since each implemented loss gradient is averaged over examples, the Batch GD gradients are mean gradients. See [`model/gradient_descent.py`](model/gradient_descent.py).

| Method | Update | Strength | Trade-off |
| --- | --- | --- | --- |
| **Batch GD** | One vectorized update after computing gradients over all `m` rows | Stable, smooth progress | Each update costs a full pass over the dataset. |
| **SGD** | Update once per shuffled row, using that row's gradient | Frequent, inexpensive updates; can begin improving quickly | Noisy loss trajectory and more sensitivity to learning rate. |

SGD shuffles row indices at each epoch using a seeded NumPy random generator. Both methods record the selected loss over the full dataset after each epoch. Training and preprocessing are in [`model/train_numpy.py`](model/train_numpy.py).

## Interview Quick Answers

- **Why standardize features before Gradient Descent?** Features on very different scales create unbalanced gradients and make one learning rate hard to use. Standardization improves conditioning and makes updates more balanced.
- **Why standardize the target too?** It keeps prediction residuals and gradients at a manageable scale. Convert predictions back with `price = standardized_prediction × target_scale + target_mean`.
- **Does Linear Regression have local minima?** With a linear model and a convex loss such as MSE, the objective is convex in `w` and `b`; every local minimum is global. A poor learning rate can still prevent convergence.
- **What if the learning rate is too large or too small?** Too large can overshoot, oscillate, or diverge. Too small converges slowly. Start with a modest value, inspect the loss curve, and tune against validation data.
- **When choose MSE, MAE, or Huber?** Use MSE when large errors should be strongly penalized; MAE when outlier resistance matters; Huber when you want MSE-like behavior near zero and less influence from extreme residuals.
- **What does this project's loss history tell you?** It shows optimization behavior on training data, not generalization. A validation split is needed to assess performance on unseen houses.

## Run the NumPy Trainer

From the repository root with Python and NumPy installed:

```bash
python -m model.train_numpy --loss mse --method batch --learning-rate 0.03 --epochs 500
```

- Loss options: `mse`, `mae`, `huber`.
- Optimizer options: `batch`, `sgd`.
- Output: `model/linear_regression_numpy.json`, containing fitted parameters, preprocessing values, and per-epoch training loss.
- The current trainer reports training loss only; it does not calculate validation metrics.
