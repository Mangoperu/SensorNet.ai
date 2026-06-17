"""
Reusable data-loading utilities for the TEP fault diagnosis project.

Keeping this logic out of the notebooks means you can reuse it in a script,
a FastAPI service (Phase 7), or a unit test without copy-pasting notebook cells.
"""

import pandas as pd


def load_tep_data(path: str) -> pd.DataFrame:
    """Load a TEP dataset CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)
    return df


def load_and_combine(normal_path: str, faulty_path: str) -> pd.DataFrame:
    """Load and combine separate fault-free and faulty TEP files.

    Use this if your TEP download came as two separate files rather than
    one combined CSV.
    """
    df_normal = pd.read_csv(normal_path)
    df_faulty = pd.read_csv(faulty_path)
    return pd.concat([df_normal, df_faulty], ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of process variable columns (xmeas_* and xmv_*)."""
    return [c for c in df.columns if c.startswith("xmeas_") or c.startswith("xmv_")]
