"""Shared paths, config and logging for the backtest phases."""
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
CHARTS = RESULTS / 'charts'
CACHE = ROOT / '.cache'

# Screens in report order: key -> (label, kind, parameter)
SCREENS = {
    'cluster3': ('Cluster buys (≥3 insiders)', 'cluster', 3),
    'exec200k': ('Top executive buys (≥$200K)', 'exec', 200_000),
    'exec500k': ('Top executive buys (≥$500K)', 'exec', 500_000),
    'cluster5': ('Tighter clusters (≥5 insiders)', 'cluster', 5),
    'cluster10': ('Tighter clusters (≥10 insiders)', 'cluster', 10),
}


def load_config():
    with open(ROOT / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['sec']['user_agent'] = os.environ.get('SEC_USER_AGENT', cfg['sec']['user_agent'])
    return cfg


def get_logger(name):
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout),
                                      logging.FileHandler(ROOT / 'backtest.log')])
    return logger


def save_table(df, path_stem, index=False):
    """Write a DataFrame as both CSV and Parquet."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix('.csv'), index=index)
    out = df.reset_index() if index else df
    # Parquet needs plain column names and no mixed object columns
    out = out.copy()
    out.columns = [str(c) for c in out.columns]
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype('string')
    out.to_parquet(path_stem.with_suffix('.parquet'), index=False)
