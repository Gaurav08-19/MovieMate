"""
Data loading utilities for the MovieMate chatbot.
Supports loading from local CSV or fetching from the OMDb public API.
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "movies.csv"


def load_movies(csv_path: str = None) -> pd.DataFrame:
    """Load the movie dataset from a CSV file.

    Args:
        csv_path: Path to a CSV file. Defaults to data/movies.csv.

    Returns:
        A cleaned pandas DataFrame with movie records.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run the data collection script or place a movies.csv in the data/ directory."
        )

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} movies from {path}")
    return df


def fetch_from_omdb(titles: list, api_key: str) -> pd.DataFrame:
    """Fetch movie data from the OMDb API for a list of titles.

    Args:
        titles: List of movie title strings.
        api_key: Your OMDb API key (free tier at omdbapi.com).

    Returns:
        DataFrame with fetched movie records.
    """
    records = []
    base_url = "http://www.omdbapi.com/"

    for title in titles:
        params = {"t": title, "apikey": api_key, "plot": "short"}
        try:
            response = requests.get(base_url, params=params, timeout=5)
            data = response.json()
            if data.get("Response") == "True":
                records.append({
                    "title": data.get("Title", ""),
                    "year": int(data.get("Year", "0")[:4]) if data.get("Year") else None,
                    "rating": float(data.get("imdbRating", 0)) if data.get("imdbRating") != "N/A" else None,
                    "votes": data.get("imdbVotes", "0").replace(",", ""),
                    "genre": data.get("Genre", "").replace(", ", "|"),
                    "director": data.get("Director", ""),
                    "cast": data.get("Actors", ""),
                    "duration_mins": int(data.get("Runtime", "0 min").split()[0]) if data.get("Runtime") else None,
                    "plot_summary": data.get("Plot", ""),
                    "language": data.get("Language", "English"),
                })
        except Exception as e:
            print(f"  Warning: Could not fetch '{title}': {e}")

    df = pd.DataFrame(records)
    print(f"Fetched {len(df)} movies from OMDb API")
    return df


def save_movies(df: pd.DataFrame, path: str = None):
    """Save movies DataFrame to CSV."""
    out_path = path or str(DEFAULT_CSV)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} movies to {out_path}")
