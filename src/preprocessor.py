"""
Data preprocessing utilities for MovieMate.
Handles cleaning, normalization, and text representation for embedding.
"""

import re
import pandas as pd
import numpy as np


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw movie DataFrame.

    Steps:
    - Drop duplicate titles
    - Fill missing values with sensible defaults
    - Normalize ratings and votes
    - Parse genres into a list column
    - Normalize text fields (strip whitespace, fix encoding)

    Args:
        df: Raw movies DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Drop exact duplicates
    df.drop_duplicates(subset=["title", "year"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Strip whitespace from string columns
    str_cols = ["title", "genre", "director", "cast", "plot_summary", "language"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fill missing values
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(df["rating"].median() if "rating" in df else 7.0)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2000).astype(int)
    df["duration_mins"] = pd.to_numeric(df["duration_mins"], errors="coerce").fillna(0).astype(int)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df["plot_summary"] = df["plot_summary"].replace("nan", "No summary available.")
    df["language"] = df["language"].replace("nan", "English")

    # Parse genres into a Python list (stored as string in CSV)
    df["genre_list"] = df["genre"].apply(
        lambda g: [x.strip() for x in re.split(r"[|,/]", str(g))] if pd.notna(g) else []
    )

    # Parse cast into a list (top 3 actors)
    df["cast_list"] = df["cast"].apply(
        lambda c: [x.strip() for x in str(c).split(",")][:3] if pd.notna(c) else []
    )

    # Add a decade column for EDA
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"

    print(f"Preprocessing complete: {len(df)} movies retained.")
    return df


def build_text_representation(row: pd.Series) -> str:
    """Build a rich text string for a single movie row, used for embedding.

    Combines all relevant fields into a single document that captures
    the semantic content of the movie.

    Args:
        row: A single DataFrame row (movie record).

    Returns:
        A formatted string representation of the movie.
    """
    genre_str = ", ".join(row["genre_list"]) if isinstance(row.get("genre_list"), list) else str(row.get("genre", ""))
    cast_str = ", ".join(row["cast_list"]) if isinstance(row.get("cast_list"), list) else str(row.get("cast", ""))

    text = (
        f"Title: {row['title']} ({row['year']})\n"
        f"Genre: {genre_str}\n"
        f"Director: {row['director']}\n"
        f"Cast: {cast_str}\n"
        f"Rating: {row['rating']}/10 | Duration: {row['duration_mins']} mins\n"
        f"Language: {row.get('language', 'English')}\n"
        f"Plot: {row['plot_summary']}"
    )
    return text


def add_text_representations(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'text' column with the full text representation for each movie."""
    df = df.copy()
    df["text"] = df.apply(build_text_representation, axis=1)
    return df
