# train_and_save_models.py
import joblib
import torch
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from prophet import Prophet

# Example LightGBM
lgb_model = LGBMRegressor()
lgb_model.fit(X_train, y_train)
joblib.dump(lgb_model, "models/lightgbm.pkl")

# Example XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "models/xgboost.pkl")

# Example Prophet
prophet_df = df.rename(columns={"date": "ds", "price": "y"})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
joblib.dump(prophet_model, "models/prophet.pkl")

# Example LSTM
torch.save(lstm_model.state_dict(), "models/lstm.pt")
