import os
from datetime import datetime
import subprocess

def should_retrain(filepath, max_age_days=1):
    """Check if the forecast CSV is older than max_age_days"""
    if not os.path.exists(filepath):
        return True
    modified = datetime.fromtimestamp(os.path.getmtime(filepath))
    age = (datetime.now() - modified).days
    return age >= max_age_days

def trigger_colab_training(ticker):
    """
    Optional: trigger Colab training manually.
    Here we just open a URL or run a shell command if you have a Colab API.
    """
    # Example: print message to remind user
    print(f"Please run Colab training for {ticker} to refresh data.")
    # OR, if you have local scripts to trigger Colab, you can call them
    # subprocess.run(["python", "train_sp500.py", "--ticker", ticker])
