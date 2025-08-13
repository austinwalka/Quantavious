#!/usr/bin/env python3
"""
Train full meta models for a list of tickers. This may take time depending on samples & paths.
Usage:
  python scripts/train_meta_full.py AAPL MSFT --start 2016-01-01 --samples 200
"""
import sys
import argparse
from src.meta_learner import build_and_train_full_meta_for_ticker

parser = argparse.ArgumentParser()
parser.add_argument('tickers', nargs='+')
parser.add_argument("--start", default="2016-01-01")
parser.add_argument("--trading_days", type=int, default=15)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--n_paths", type=int, default=200)
parser.add_argument("--max_samples", type=int, default=200)
args = parser.parse_args()

for t in args.tickers:
    print("Training meta for", t)
    model, df = build_and_train_full_meta_for_ticker(t, start=args.start, trading_days=args.trading_days, step=args.step, n_paths=args.n_paths, max_samples=args.max_samples)
    print("Saved meta model for", t, "samples:", len(df))
