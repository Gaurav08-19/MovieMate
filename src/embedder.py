"""
Embedding generation for MovieMate.
Uses sentence-transformers to convert movie text representations into dense vectors.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Lazy import to avoid import errors if not installed
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback: TF-IDF based embeddings (always available via sklearn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIR = Path(__file__).parent.parent / "data" / "embeddings"


class MovieEmbedder:
    """Generates and caches vector embeddings for movie text representations.

    Prefers sentence-transformers (semantic, high quality). Falls back to
    TF-IDF if sentence-transformers is not installed.

    Args:
        use_tfidf: Force TF-IDF mode (useful for quick testing without GPU/heavy deps).
        model_name: HuggingFace model name for sentence-transformers.
    """

    def __init__(self, use_tfidf: bool = False, model_name: str = MODEL_NAME):
        self.use_tfidf = use_tfidf or not SENTENCE_TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        self.embeddings = None
        self.dimension = None

        if self.use_tfidf:
            print("Using TF-IDF embeddings (install sentence-transformers for semantic search).")
        else:
            print(f"Using sentence-transformers model: {model_name}")

    def _load_model(self):
        if self.use_tfidf:
            if self._tfidf is None:
                self._tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        else:
            if self._model is None:
                print(f"Loading model '{self.model_name}' (first run downloads ~90MB)...")
                self._model = SentenceTransformer(self.model_name)

    def fit_transform(self, texts: list) -> np.ndarray:
        """Generate embeddings for a list of text strings.

        For TF-IDF mode, also fits the vectorizer on the corpus.

        Args:
            texts: List of text strings (one per movie).

        Returns:
            2D numpy array of shape (n_movies, embedding_dim).
        """
        self._load_model()

        if self.use_tfidf:
            matrix = self._tfidf.fit_transform(texts).toarray().astype("float32")
            embeddings = normalize(matrix, norm="l2")
        else:
            print("Generating embeddings (this may take a minute on first run)...")
            embeddings = self._model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
            )

        self.embeddings = embeddings.astype("float32")
        self.dimension = self.embeddings.shape[1]
        print(f"Generated embeddings: shape={self.embeddings.shape}")
        return self.embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a vector.

        Args:
            query: User's natural language query.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        self._load_model()

        if self.use_tfidf:
            vec = self._tfidf.transform([query]).toarray().astype("float32")
            vec = normalize(vec, norm="l2")
            return vec[0]
        else:
            return self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

    def save(self, path: str = None):
        """Persist embeddings to disk as a .npy file."""
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Call fit_transform first.")
        save_path = path or str(EMBED_DIR / "embeddings.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.embeddings)
        print(f"Embeddings saved to {save_path}")

    def load(self, path: str = None):
        """Load previously saved embeddings from disk."""
        load_path = path or str(EMBED_DIR / "embeddings.npy")
        self.embeddings = np.load(load_path)
        self.dimension = self.embeddings.shape[1]
        print(f"Loaded embeddings from {load_path}: shape={self.embeddings.shape}")
        return self.embeddings
