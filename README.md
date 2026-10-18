# Melbourne Housing Price Prediction

A compact, notebook-based machine learning project that explains and implements Linear Regression with NumPy from scratch using Melbourne housing data.

## Project files

- [`melbourne_housing_ml.ipynb`](melbourne_housing_ml.ipynb) contains the complete workflow: data inspection, train/test split, feature scaling, one-hot encoding, MSE/MAE/Huber losses and gradients, Batch Gradient Descent and SGD, learning curves, and held-out price metrics.
- [`Resources/clean_melbourne_housing.csv`](Resources/clean_melbourne_housing.csv) is the cleaned dataset used by the notebook (17,698 rows).

There is no web app or API to configure. The notebook is the project and model implementation; the CSV remains a separate data file so it can be loaded directly without embedding several megabytes in the notebook.

## Run it

From the repository folder, install the notebook dependencies and start Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install numpy pandas matplotlib jupyter
jupyter lab
```

Open `melbourne_housing_ml.ipynb` and run all cells in order. It expects to be run from the repository folder so it can read `Resources/clean_melbourne_housing.csv`.

The data is split into training and test sets. Scaling statistics are computed from training data only. The final MAE and RMSE are reported in Australian dollars on held-out rows; these are educational metrics, not formal property valuations.
