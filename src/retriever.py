"""
FAISS-based vector similarity retrieval for MovieMate.
Supports both flat (exact) and IVF (approximate) indexes.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Falling back to numpy dot-product search.")

INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"


class MovieRetriever:
    """Retrieves the most relevant movies for a query using vector similarity search.

    Uses FAISS for efficient nearest-neighbor search. Falls back to a pure
    numpy implementation if FAISS is not installed.

    Args:
        df: Preprocessed movies DataFrame (with 'text' column).
        embedder: A fitted MovieEmbedder instance.
    """

    def __init__(self, df: pd.DataFrame, embedder):
        self.df = df.reset_index(drop=True)
        self.embedder = embedder
        self.index = None
        self._embeddings_matrix = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, embeddings: np.ndarray = None):
        """Build the search index from movie embeddings.

        Args:
            embeddings: Pre-computed embeddings array. If None, uses
                        embedder.embeddings (must call fit_transform first).
        """
        if embeddings is None:
            if self.embedder.embeddings is None:
                raise RuntimeError("Call embedder.fit_transform(texts) before building the index.")
            embeddings = self.embedder.embeddings

        self._embeddings_matrix = embeddings.astype("float32")
        dim = self._embeddings_matrix.shape[1]

        if FAISS_AVAILABLE:
            # IndexFlatIP: exact inner-product search (= cosine similarity for L2-normalized vectors)
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self._embeddings_matrix)
            print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        else:
            print(f"Numpy index built: {len(self._embeddings_matrix)} vectors, dim={dim}")

    def save_index(self, path: str = None):
        """Persist the FAISS index to disk."""
        if not FAISS_AVAILABLE or self.index is None:
            return
        save_path = path or str(INDEX_DIR / "movies.index")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(self.index, save_path)
        print(f"FAISS index saved to {save_path}")

    def load_index(self, path: str = None):
        """Load a previously saved FAISS index."""
        if not FAISS_AVAILABLE:
            return
        load_path = path or str(INDEX_DIR / "movies.index")
        self.index = faiss.read_index(load_path)
        print(f"FAISS index loaded from {load_path}: {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant movies for a natural language query.

        Args:
            query: User's natural language query string.
            k: Number of results to return.

        Returns:
            List of dicts, each containing movie metadata + similarity score.
        """
        query_vec = self.embedder.encode_query(query).astype("float32").reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_vec, k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
        else:
            # Numpy fallback: cosine similarity via dot product (vectors are L2-normalized)
            sims = self._embeddings_matrix.dot(query_vec.T).flatten()
            top_idx = np.argsort(sims)[::-1][:k]
            indices = top_idx.tolist()
            scores = sims[top_idx].tolist()

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx].to_dict()
            row["similarity_score"] = round(float(score), 4)
            results.append(row)

        return results

    def search_by_filter(
        self,
        query: str = "",
        genre: str = None,
        min_rating: float = None,
        max_year: int = None,
        min_year: int = None,
        director: str = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: apply metadata filters then rank by vector similarity.

        Args:
            query: Natural language query (optional, used for ranking).
            genre: Filter by genre string (case-insensitive partial match).
            min_rating: Minimum IMDb rating.
            max_year: Maximum release year.
            min_year: Minimum release year.
            director: Filter by director name (partial match).
            k: Number of results.

        Returns:
            Filtered and ranked list of movie dicts.
        """
        filtered = self.df.copy()

        if genre:
            filtered = filtered[
                filtered["genre"].str.contains(genre, case=False, na=False)
            ]
        if min_rating is not None:
            filtered = filtered[filtered["rating"] >= min_rating]
        if min_year is not None:
            filtered = filtered[filtered["year"] >= min_year]
        if max_year is not None:
            filtered = filtered[filtered["year"] <= max_year]
        if director:
            filtered = filtered[
                filtered["director"].str.contains(director, case=False, na=False)
            ]

        if filtered.empty:
            return []

        # If we have a query, rank the filtered subset by similarity
        if query and self._embeddings_matrix is not None:
            query_vec = self.embedder.encode_query(query).astype("float32")
            subset_embeddings = self._embeddings_matrix[filtered.index]
            sims = subset_embeddings.dot(query_vec)
            top_idx = np.argsort(sims)[::-1][:k]
            results = []
            for i in top_idx:
                row = filtered.iloc[i].to_dict()
                row["similarity_score"] = round(float(sims[i]), 4)
                results.append(row)
            return results
        else:
            return filtered.head(k).to_dict("records")
