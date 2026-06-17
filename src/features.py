"""
Feature engineering utilities for the TEP fault diagnosis project.

All functions operate per-simulationRun so that no information leaks
across the boundary between two different simulation runs.
"""

import pandas as pd


def add_engineered_features(
    data: pd.DataFrame,
    feature_cols: list,
    run_col: str = "simulationRun",
    sample_col: str = "sample",
    window: int = 5,
) -> pd.DataFrame:
    """Add rolling mean, rolling std, and first-difference features.

    Computed within each simulationRun group so no information leaks
    across run boundaries.

    Parameters
    ----------
    data : pd.DataFrame
    feature_cols : list of str
        Columns to engineer features from (e.g. xmeas_*, xmv_*).
    run_col : str
        Column identifying which simulation run a row belongs to.
    sample_col : str
        Column identifying the time step within a run.
    window : int
        Rolling window size (in samples).

    Returns
    -------
    pd.DataFrame with the original columns plus _rollmean, _rollstd, _diff
    suffixed engineered columns.
    """
    data = data.sort_values([run_col, sample_col]).copy()
    grouped = data.groupby(run_col)[feature_cols]

    roll_mean = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    roll_std = grouped.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
    diff = grouped.transform(lambda s: s.diff().fillna(0))

    roll_mean.columns = [f"{c}_rollmean" for c in feature_cols]
    roll_std.columns = [f"{c}_rollstd" for c in feature_cols]
    diff.columns = [f"{c}_diff" for c in feature_cols]

    return pd.concat([data, roll_mean, roll_std, diff], axis=1)
