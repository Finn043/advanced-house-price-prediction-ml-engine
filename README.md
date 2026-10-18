# Melbourne Housing Price Prediction — NumPy Learning Engine

A Melbourne property-price project built around the repository's cleaned housing CSV. The original repository includes exploratory analysis and a small Flask prediction page. This version adds a transparent linear regression engine implemented with NumPy: the loss functions, gradients, and batch/SGD parameter updates are written explicitly so the training process can be studied rather than hidden behind a machine-learning estimator.

> **Scope:** The NumPy engine is a learning and demonstration implementation. Its JSON artifact is produced by the trainer, while the existing Flask page still loads the repository's separate `model/model.pkl` artifact. The two prediction paths are not connected.

## What is in this project

- A from-scratch linear regression optimizer using vectorized NumPy operations.
- MSE, MAE, and Huber losses, including derivatives with respect to predictions.
- Batch Gradient Descent and single-example Stochastic Gradient Descent (SGD).
- A CSV trainer that standardizes numeric features and target values, one-hot encodes categorical features, and saves weights plus loss history to JSON.
- The original Melbourne dataset, analysis notebook, plots, and Flask/Jinja prediction page from the repository.

## Dataset and feature preparation

The trainer reads `Resources/clean_melbourne_housing.csv` directly and does not modify the CSV or require a database. It predicts `Price` from:

| Feature | Preparation |
| --- | --- |
| `Rooms`, `Distance`, `Bathroom`, `Car` | Standardized using the dataset mean and standard deviation |
| `Type`, `Region` | One-hot encoded using the category values present in the CSV |
| `Price` | Standardized as the training target; mean and standard deviation are saved for reversing the scaling |

The trainer validates required columns, finite numeric values, positive prices, and a minimum of two rows. The current implementation fits and reports training loss on the full dataset; it does not calculate validation or test metrics.

## Project layout

```text
.
├── Resources/
│   └── clean_melbourne_housing.csv   # Dataset used by the NumPy trainer
├── model/
│   ├── loss_functions.py             # MSE, MAE, Huber and prediction gradients
│   ├── gradient_descent.py           # Batch GD and per-example SGD updates
│   ├── linear_regression.py          # Linear model using the optimizer above
│   ├── train_numpy.py                # Preprocessing, training and JSON export
│   └── model.pkl                     # Existing artifact used by the Flask page
├── static/                           # Existing page styles and analysis figures
├── templates/                        # Existing Flask/Jinja page
├── app.py                            # Existing web prediction page (pickle model)
├── main.py                           # Original analysis script
├── melbourne_housing_data.ipynb       # Original analysis notebook
└── requirements.txt                  # Original app / analysis dependencies
```

## Training the NumPy model

The trainer only needs NumPy. From the repository root, create an environment and install a current NumPy release:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install "numpy>=1.24,<3"
python -m model.train_numpy --loss mse --method batch --learning-rate 0.03 --epochs 500
```

The command prints the row count and final standardized training loss. The default output is `model/linear_regression_numpy.json`; this generated artifact is ignored by Git. Choose among the supported options:

```bash
python -m model.train_numpy --loss mae --method sgd --learning-rate 0.001 --epochs 20
python -m model.train_numpy --loss huber --method batch --learning-rate 0.03 --epochs 500
```

`--method sgd` updates the model after each randomly shuffled training row. Its learning rate usually needs to be smaller than the batch method's. The default shuffle seed is fixed, so runs with the same inputs and settings are repeatable.

The JSON artifact contains feature names, category vocabularies, scaling values, fitted weights and bias, selected optimizer/loss, row count, and per-epoch training loss. Predictions in standardized target units can be converted back to the dataset's price units with:

```text
price = standardized_prediction * target_scale + target_mean
```

The trainer currently saves training history but does not expose a training API, draw a learning-curve chart, or serve predictions from the new JSON artifact. The Flask page remains a separate legacy path backed by `model/model.pkl`.

## How the learning engine works

### 1. Linear model

For a feature vector with `n` components, the model predicts:

\[
\hat y_i = \mathbf{x}_i^T\mathbf{w} + b
\]

Here, `w` contains one coefficient per prepared feature and `b` is the intercept. In matrix form, predictions for `m` rows are `Xw + b`.

### 2. Loss and its gradient

Let the residual be `eᵢ = ŷᵢ − yᵢ`. The code computes the mean loss and its derivative with respect to each prediction. The optimizer then propagates that derivative through the linear model.

**Mean Squared Error (MSE)**

\[
J_{MSE}=\frac{1}{m}\sum_{i=1}^{m}e_i^2,
\qquad
\frac{\partial J}{\partial\hat y_i}=\frac{2e_i}{m}
\]

Squaring makes large residuals contribute disproportionately. This gives MSE smooth gradients, but also makes it sensitive to outliers.

**Mean Absolute Error (MAE)**

\[
J_{MAE}=\frac{1}{m}\sum_{i=1}^{m}|e_i|,
\qquad
\frac{\partial J}{\partial\hat y_i}=\frac{\operatorname{sign}(e_i)}{m}
\]

MAE grows linearly with error size, so outliers have less influence than under MSE. At zero residual, the absolute value has no unique derivative; the implementation uses `numpy.sign(0) = 0`, a valid subgradient choice.

**Huber loss** (threshold `δ = 1` in standardized target units)

\[
L_\delta(e)=
\begin{cases}
\frac{1}{2}e^2 & |e|\leq\delta\\
\delta(|e|-\frac{1}{2}\delta) & |e|>\delta
\end{cases}
\]

Its derivative with respect to prediction is `clip(e, -δ, δ) / m`. Huber is quadratic near zero and linear for large residuals, combining smooth behavior near a fit with reduced sensitivity to extreme errors.

### 3. Gradient Descent updates

For batch Gradient Descent, compute the prediction gradient for the entire dataset, then use the chain rule:

\[
\nabla_{\mathbf w}J = X^T\nabla_{\hat y}J,
\qquad
\frac{\partial J}{\partial b}=\sum_{i=1}^{m}\frac{\partial J}{\partial\hat y_i}
\]

Update the coefficients and intercept in the negative-gradient direction:

\[
\mathbf w \leftarrow \mathbf w - \alpha\nabla_{\mathbf w}J,
\qquad
b \leftarrow b - \alpha\frac{\partial J}{\partial b}
\]

`α` is the learning rate. In this implementation the mean-loss derivative already includes division by `m`, so the batch gradients are averaged over the dataset.

**Batch mode** performs one vectorized update after seeing every row. **SGD mode** shuffles row indices and applies one update per row using that row's gradient; after each epoch, both methods record the loss evaluated over the full dataset. SGD therefore records a comparable full-data history while its individual updates are noisier.

The features and target are standardized to keep their scales manageable for optimization. This makes a single learning-rate setting more practical and means the reported loss is in standardized target units, not dollars.

### 4. Convergence and limitations

A learning rate that is too large can cause the loss to oscillate or grow; one that is too small can make progress very slow. Inspect the saved `history` values to see whether loss is decreasing. The optimizer has no early stopping, regularization, learning-rate schedule, or automatic divergence recovery. Since the current trainer uses the full dataset for fitting and loss reporting, its history is not an estimate of performance on unseen houses.

## Existing Flask page

The repository's original page can be launched with the dependencies and Python versions compatible with its pinned packages:

```bash
python -m pip install -r requirements.txt
python app.py
```

It loads `model/model.pkl` and renders `templates/index.html`. Training the NumPy model does not replace that pickle file or change the page's predictions. The notebook and `main.py` are retained as original analysis materials; the `model/train_numpy.py` command above is the reproducible entry point for the new from-scratch trainer.

## Interview Q&A

**How does Gradient Descent optimize Linear Regression?**

It calculates how the chosen cost changes with each coefficient and the intercept, then updates each parameter in the opposite direction of its gradient. Repeating these updates lowers the cost when the learning rate is appropriate.

**What is the role of the learning rate?**

It controls the size of each parameter update. A very large value can overshoot and diverge; a very small value converges slowly. Feature scaling helps make gradient magnitudes and learning rates easier to manage.

**Why is MSE more sensitive to outliers than MAE?**
MSE squares residuals, so a residual ten times larger contributes one hundred times as much squared error. MAE's contribution only grows tenfold. MSE is smooth and convenient to optimize; MAE is more robust to large errors but is not differentiable at zero.

**What is the difference between Batch Gradient Descent and SGD?**

Batch GD calculates one gradient from all training rows per update. SGD updates after each row, so it can make frequent progress but its path is noisier and depends on row order. This project shuffles the rows each epoch with a seeded random generator.

**Why standardize features?**

Features with very different numeric scales can produce poorly scaled gradients. Standardization makes optimization steps more balanced. The means and scales must be retained so inference uses the same transformation.

**What is overfitting?**

Overfitting occurs when a model fits patterns specific to training data, including noise, and performs worse on unseen data. Detect it by comparing training and validation performance. This trainer currently reports only training loss, so it cannot by itself diagnose generalization or overfitting.

**Does Linear Regression have local minima?**

With a convex loss such as MSE and a linear model, the objective is convex in the coefficients. Any local minimum is therefore a global minimum. Poor learning-rate choices can still prevent Gradient Descent from reaching it effectively.

**Why include Huber loss?**

Huber behaves like MSE for small errors and like MAE for large errors. It provides smooth gradients near zero while reducing the influence of large residuals.

**What does the saved loss history show?**

It records the selected loss on the full training dataset after each epoch. It is useful for checking optimization behavior, but it is not a validation curve and does not establish performance on unseen data.

## Credits and source

The original Melbourne housing data exploration and web application were cloned from [yawavi92/ML-Housing_price_predection](https://github.com/yawavi92/ML-Housing_price_predection). The NumPy loss, optimization, linear regression, and training modules were added in this project to make the core learning steps explicit.
