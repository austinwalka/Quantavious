# src/ensemble.py
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Iterable, Tuple

# finBERT cache loader (if you want to pre-load HF model, do it once).
@st.cache_resource(show_spinner=False)
def load_finbert_pipeline(model_name: str = "ProsusAI/finbert"):
    """
    Loads finBERT via HuggingFace pipeline (cached by Streamlit).
    If HF is not available or too heavy, caller should catch exceptions and fallback.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True)
        return clf
    except Exception as e:
        st.warning(f"Could not load finBERT pipeline: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=60*60)  # cache forecasts for 1 hour
def cache_forecast_dataframe(ticker: str, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Store a forecast DataFrame keyed by ticker (streamlit cache wrapper)."""
    # This function simply returns the dataframe and allows Streamlit to cache it.
    return forecast_df.copy()


def compute_model_skill(predictions: Dict[str, np.ndarray], actual: np.ndarray) -> Dict[str, float]:
    """
    Compute simple per-model RMSE skill on a provided holdout actual vector.
    predictions: dict name -> np.array (aligned to actual)
    returns dict name -> rmse
    """
    skills = {}
    for name, arr in predictions.items():
        try:
            rmse = float(np.sqrt(np.mean((np.array(arr[:len(actual)]) - np.array(actual[:len(arr)]))**2)))
        except Exception:
            rmse = np.inf
        skills[name] = rmse
    return skills


def default_weights_from_skill(skills: Dict[str, float], epsilon: float = 1e-6) -> Dict[str, float]:
    """
    Convert RMSE skills into normalized weights: weight_i ~ 1 / (rmse_i + epsilon)
    Returns weights summing to 1.
    """
    inv = {k: 1.0 / (v + epsilon) if np.isfinite(v) and v > 0 else 0.0 for k, v in skills.items()}
    s = sum(inv.values())
    if s <= 0:
        # fallback equal weights
        n = len(skills)
        return {k: 1.0 / n for k in skills}
    return {k: float(inv[k] / s) for k in skills}


def sentiment_tilt_weights(base_weights: Dict[str, float],
                           sentiment: float,
                           trend_models: Iterable[str] = ("GBM", "ML"),
                           meanrev_models: Iterable[str] = ("OU",),
                           tilt_strength: float = 0.5) -> Dict[str, float]:
    """
    Tilt base_weights according to sentiment (-1..1).
    - Positive sentiment -> favor trend_models
    - Negative sentiment -> favor meanrev_models
    tilt_strength in [0..1] controls max shift fraction (0.5 means up to 50% reallocation)
    The function returns normalized weights summing to 1.
    """
    # copy
    w = base_weights.copy()
    # compute total mass to move
    tilt = float(np.clip(sentiment, -1.0, 1.0)) * float(np.clip(tilt_strength, 0.0, 1.0))
    if abs(tilt) < 1e-8:
        return w

    # determine recipients and donors
    recipients = set(trend_models) if tilt > 0 else set(meanrev_models)
    donors = set(meanrev_models) if tilt > 0 else set(trend_models)

    # compute donor mass and recipient capacity
    donor_mass = sum(w.get(d, 0.0) for d in donors)
    recipient_mass = sum(w.get(r, 0.0) for r in recipients)
    if donor_mass <= 0 or recipient_mass <= 0:
        return w

    # amount to shift = donor_mass * |tilt|
    shift = donor_mass * abs(tilt)
    # distribute shift proportionally among recipients, subtract proportionally from donors
    # donors scale down proportionally
    for d in donors:
        if d in w:
            w[d] = max(0.0, w[d] - (w[d] / donor_mass) * shift)
    for r in recipients:
        if r in w:
            w[r] = w[r] + (w[r] / recipient_mass) * shift

    # renormalize
    s = sum(w.values())
    if s <= 0:
        # equal fallback
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: float(v / s) for k, v in w.items()}


def blend_paths(paths: Dict[str, np.ndarray],
                weights: Dict[str, float]) -> np.ndarray:
    """
    Combine per-model paths into a single blended path using positional weighted average.
    paths: dict model -> 1D np.array (length T) or 2D (n_paths x T). We'll accept 1D here.
    weights: dict model -> float
    Returns 1D array length T
    """
    # find T
    arrays = {k: np.asarray(v).flatten() for k, v in paths.items()}
    lengths = [arr.shape[0] for arr in arrays.values()]
    T = min(lengths)
    # weighted sum
    total = np.zeros(T, dtype=float)
    for k, arr in arrays.items():
        w = weights.get(k, 0.0)
        total += w * arr[:T]
    return total


def make_best_guess(predictions: Dict[str, np.ndarray],
                    holdout_actual: np.ndarray = None,
                    sentiment: float = 0.0,
                    tilt_strength: float = 0.5) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Main helper to compute blended Best_Guess path and return (best_path, final_weights)
    - predictions: dict of model_name -> 1D np.array (path of predicted prices aligned by time)
    - holdout_actual: optional array of actuals to compute skill for weighting
    - sentiment: finBERT score in [-1,1] used to tilt weights toward trend or mean-rev
    - tilt_strength: how strongly sentiment may reallocate weights (0..1)
    """
    # compute skill weights if holdout_actual present else equal weights
    if holdout_actual is not None:
        skill = compute_model_skill(predictions, np.asarray(holdout_actual))
        base = default_weights_from_skill(skill)
    else:
        n = max(1, len(predictions))
        base = {k: 1.0 / n for k in predictions.keys()}

    # sentiment tilt: treat GBM and ML as trend, OU as meanrev
    final_w = sentiment_tilt_weights(base, sentiment,
                                    trend_models=("GBM", "ML", "ML_Path"),
                                    meanrev_models=("OU", "Langevin"),
                                    tilt_strength=tilt_strength)

    best = blend_paths(predictions, final_w)
    return best, final_w
