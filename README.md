# Melbourne House Price Prediction

This project explores how a house-price prediction model learns from data. Its core is a linear regression engine written with NumPy: given a home's features, it produces an estimated price, measures how far that estimate is from the known price, and adjusts its parameters to improve future predictions.

The goal is to make that learning process visible in code. The model, loss functions, and gradient descent updates are implemented directly rather than delegated to a machine-learning estimator.

## How a prediction is made

The trainer uses `Resources/clean_melbourne_housing.csv`. For each house, it prepares these inputs:

- Numeric: `Rooms`, `Distance`, `Bathroom`, and `Car`.
- Categorical: `Type` and `Region`, converted into one-hot features so the model can work with them numerically.

Numeric features are standardized so that values with different units and ranges are on comparable scales. The target `Price` is standardized during training too. The model combines the prepared features with learned weights and a bias to produce a predicted, standardized price. The saved target mean and scale can convert that prediction back to the dataset's price units.

In short, each feature has a learned contribution to the prediction. The model starts with initial weights and bias, then learns values that better fit the examples in the dataset.

## How the model learns

Training repeats three steps:

1. **Predict:** Use the current weights and bias to estimate prices for the training houses.
2. **Measure error:** Compare predictions with the known prices using a loss function. Lower loss means predictions are closer overall under that loss's definition.
3. **Update:** Gradient Descent calculates how each weight and the bias contributed to the error, then nudges them in a direction that should reduce the loss.

This process runs for the requested number of epochs. An epoch is one training round. The trainer records the full training-set loss after every epoch in the output JSON, so you can inspect whether learning is progressing. The loss history is measured on training data; it does not show how well the model predicts houses it has never seen.

### Loss options

The loss function defines what kinds of prediction errors the model should care about:

- **MSE (Mean Squared Error):** Squares each error before averaging. Large errors count much more heavily, which makes MSE sensitive to outliers. Its smooth gradient is useful for optimization.
- **MAE (Mean Absolute Error):** Averages the absolute size of errors. Its penalty grows steadily, so unusually large errors have less influence than with MSE. At a perfect prediction, the absolute-value loss has no unique derivative; this implementation uses a valid subgradient there.
- **Huber loss:** Uses a squared penalty for small errors and an absolute-error-like penalty for large ones. It combines smooth behavior around a good fit with less sensitivity to extreme errors. The trainer uses a threshold of `1` in standardized price units.

### Gradient Descent options

Gradient Descent is the optimization method that changes the model's weights and bias to reduce the chosen loss. The learning rate controls the size of each change. If it is too high, the loss may jump around or grow; if too low, improvement can be slow.

- **Batch:** Calculates one update using all training rows. Its updates are stable and vectorized, but each update requires a pass over the dataset.
- **SGD:** Shuffles the rows and updates after each individual example. It makes many small updates per epoch, but those updates are noisier. A fixed random seed makes the shuffle repeatable.

The optimizer records loss over the full dataset at the end of each epoch for either method. See [`model/gradient_descent.py`](model/gradient_descent.py) for the update logic and [`model/loss_functions.py`](model/loss_functions.py) for the loss calculations.

## Project files

```text
Resources/clean_melbourne_housing.csv  Dataset used by the NumPy trainer
model/
  train_numpy.py                        Loads data, prepares features, trains, saves results
  linear_regression.py                  Linear model: predictions from features, weights, and bias
  gradient_descent.py                   Batch and SGD parameter updates
  loss_functions.py                     MSE, MAE, Huber loss and prediction gradients
```

## Run training

From the project root, create a Python environment, install NumPy, and start training:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install "numpy>=1.24,<3"
python -m model.train_numpy --loss mse --method batch --learning-rate 0.03 --epochs 500
```

Choose a loss and optimization method with `--loss mse|mae|huber` and `--method batch|sgd`. For example, SGD often needs a smaller learning rate:

```bash
python -m model.train_numpy --loss mae --method sgd --learning-rate 0.001 --epochs 20
```

After training, `model/linear_regression_numpy.json` contains the learned weights and bias, feature and scaling information, selected loss and method, number of training rows, and the loss after each epoch. The generated file is ignored by Git.

## Run the web app

The React + TypeScript interface uses Tailwind CSS and communicates with a FastAPI service. The API trains the NumPy model, returns the epoch loss history for the chart, and applies the saved preprocessing values when estimating a property's price.

Install the API dependencies and start FastAPI from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

In another terminal, start the frontend:

```bash
cd frontend
npm ci
cp .env.example .env           # Windows: copy .env.example .env
npm run dev
```

Open the Vite URL printed in the terminal (usually `http://localhost:5173`). The frontend reads `VITE_API_URL` from `frontend/.env`; it defaults to `http://localhost:8000`. FastAPI serves its interactive API documentation at `http://localhost:8000/docs`.

The dashboard lets you choose MSE, MAE, or Huber loss; Batch GD or SGD; a learning rate; and an epoch count. After training, it plots the returned training-loss history and enables property price estimates using the newly trained model.

### Deploy the frontend to Netlify

`netlify.toml` sets `frontend` as the build base, runs `npm run build`, and publishes `dist`. Deploy the FastAPI service separately, then set the Netlify build environment variable `VITE_API_URL` to the public API URL. Configure the API host's `FRONTEND_ORIGIN` variable with the Netlify site origin so browser requests pass FastAPI's CORS policy.

Netlify does not run the Python API from this configuration; the backend must be reachable at the configured URL for training and prediction to work.

## What this version demonstrates

- How house data is transformed into numeric model inputs.
- How Linear Regression turns those inputs into a price estimate.
- How a loss function scores the difference between predicted and known prices.
- How Batch Gradient Descent and SGD use that error to update model parameters.
- How the selected loss changes over training epochs in the dashboard learning curve.

The trainer currently fits and reports loss on the full dataset. It does not create a validation/test split, calculate metrics on unseen houses, or claim production-grade prediction accuracy. Those steps are needed before judging how well the model generalizes.
