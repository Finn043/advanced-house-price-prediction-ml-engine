# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React, TypeScript, Vite, and Tailwind CSS are the requested frontend. FastAPI is selected for the API to connect the frontend to the NumPy trainer.

## Users

Assumption based on the project brief: the primary user is a learner or portfolio reviewer exploring how a house-price model is trained and used.

## Product Purpose

Use Melbourne housing features to train a linear regression model and estimate a property's price. Success means users can configure training, understand whether loss is decreasing, enter property details, and see a prediction from the trained model.

## Positioning

The training path exposes NumPy implementations of linear regression, loss functions, and gradient descent instead of hiding those steps behind a machine-learning estimator.

## Operating Context

Users work with the repository's cleaned Melbourne CSV, train with a selected loss and optimizer, inspect per-epoch loss, and then request a price estimate.

## Capabilities and Constraints

- The NumPy trainer uses `Resources/clean_melbourne_housing.csv` and supports MSE, MAE, Huber, Batch Gradient Descent, and SGD.
- Training currently reports loss on the training data only; it does not measure validation or test performance.
- The API and frontend are a new interface to the NumPy training path.
- The frontend must be in English, responsive, and accessible by keyboard.

## Brand Commitments

The user requested a clean, professional dashboard with subtle light/dark styling and responsive cards. This implementation uses a light editorial dashboard as the default.

## Evidence on Hand

- Dataset: `Resources/clean_melbourne_housing.csv`.
- Model code: `model/linear_regression.py`, `model/gradient_descent.py`, `model/loss_functions.py`, and `model/train_numpy.py`.
- No validation metrics or production accuracy claims are established.

## Product Principles

- Make the train → measure → update → predict workflow understandable.
- Keep prediction inputs and training controls direct and easy to scan.
- Label training loss honestly; do not present it as unseen-data accuracy.
- Keep the NumPy implementation inspectable and connected to the UI.

## Accessibility & Inclusion

Use semantic form labels, keyboard-accessible controls, readable contrast, and responsive layouts.
