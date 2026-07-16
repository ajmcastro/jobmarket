"""Loading of the raw CSVs under data/. """

from pathlib import Path

import pandas as pd

from jobmarket.config import DATA_DIR


def safe_str(value) -> str:
    return value if isinstance(value, str) and value.strip() else ""


def load_postings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "sample_postings.csv", low_memory=False)


def load_companies(data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = data_dir / "sample_companies.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def load_skills(data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = data_dir / "sample_skills.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def load_industries(data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = data_dir / "sample_industries.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else None
