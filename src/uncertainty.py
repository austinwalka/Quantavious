import numpy as np
import pandas as pd

def compute_uncertainty(predictions: pd.DataFrame):
    preds_array = predictions.values
    mean_pred = preds_array.mean(axis=1)
    std_pred = preds_array.std(axis=1)
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred
    return mean_pred, lower, upper
