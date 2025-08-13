import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def walk_forward_backtest(model_func, df, features, target, train_size=0.8):
    split = int(len(df) * train_size)
    train, test = df[:split], df[split:]
    model = model_func(train[features], train[target])
    preds = model.predict(test[features])
    rmse = np.sqrt(mean_squared_error(test[target], preds))
    return rmse
