"""Simulation Graph for a battle in a Dungeons & Dragons like game."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
import numpy as np
from openai import OpenAI


class SimpleVectorDB:
    """Simple Vector database for lightweight, in-memory applications.
    Uses dot product as a similarity metric for lookup.
    """

    def __init__(self, vectors: np.ndarray, documents: list[str], db_path: Optional[str]) -> None:
        """Initialize the SimpleVectorDB.

        Parameters
        ----------
        vectors : np.ndarray
            Pre-computed embedding vectors for the documents. Should be a 2D array
            where each row represents a document embedding.
        documents : list[str]
            List of document strings corresponding to the vectors.
        db_path : Optional[str]
            File path for persistence. If None, the database cannot be persisted.

        Notes
        -----
        The embedding dimension is inferred from the vectors shape. If vectors is empty,
        defaults to 512 dimensions.
        """
        self.db_path = db_path
        self.documents = documents
        self._embed_client = OpenAI()
        self._embed_dim = vectors.shape[-1] if vectors.size != 0 else 512
        self.vectors = vectors

    def _lookup_idx(self, query: str, n_results: int = 1) -> list[int]:
        """Return the top k results for the query, in arbitrary order.

        Private method

        Parameters
        ----------
        query : str
            The query string to search for similar documents.
        n_results : int, optional
            Number of top results to return, by default 1.

        Returns
        -------
        list[int]
            List of document indices sorted by similarity (highest first).
            The order within the top k results is arbitrary.

        Notes
        -----
        Uses dot product similarity between query embedding and document embeddings.
        """
        query_embed = self.embed(query)
        similarity = self._document_query_releveance(query_embed)
        n_results = min(n_results, len(self.documents))
        return list(k_largest_idx(similarity, n_results))

    def get_documents(self, query: str, n_results: int = 1) -> list[str]:
        """Retrieve the most similar documents for a given query in arbitrary order.

        Parameters
        ----------
        query : str
            The query string to search for similar documents.
        n_results : int, optional
            Number of top results to return, by default 1.

        Returns
        -------
        list[str]
            List of document strings that are most similar to the query,
            ordered by similarity (highest first).

        Notes
        -----
        This is a convenience method that combines lookup_idx with document retrieval.
        """
        idx = self._lookup_idx(query, n_results)
        return [self.documents[i] for i in idx]

    def embed(self, query: str) -> np.ndarray:
        """Generate embedding vector for a given text.

        Parameters
        ----------
        query : str
            The text string to embed.

        Returns
        -------
        np.ndarray
            1D numpy array containing the embedding vector of shape (embed_dim,).

        Notes
        -----
        Uses OpenAI's text-embedding-3-small model. The embedding dimension
        matches the dimension set during initialization.
        """
        response = self._embed_client.embeddings.create(
            input=query, model="text-embedding-3-small", dimensions=self._embed_dim)
        embedding = response.data[0].embedding
        return np.array(embedding)

    def add_document(self, document: str) -> None:
        """Add a new document to the vector database.

        Parameters
        ----------
        document : str
            The document text to add to the database.

        Notes
        -----
        The document is embedded using the embed method and both the document
        text and its embedding are stored. If this is the first document added
        to an empty database, it initializes the vectors array.
        """
        vector = self.embed(document)
        self.documents.append(document)
        if self.vectors.size == 0:
            self.vectors = vector.reshape(1, -1)
            return None
        self.vectors = np.concatenate([self.vectors, vector.reshape(1, -1)])

    def update_document(self, doc_idx: int, document: str) -> None:
        """Update an existing document at the specified index.

        Parameters
        ----------
        doc_idx : int
            Index of the document to update.
        document : str
            New document text to replace the existing document.

        Notes
        -----
        Both the document text and its corresponding embedding vector are updated.
        The document is re-embedded using the current embedding model.

        Raises
        ------
        IndexError
            If doc_idx is out of bounds for the documents list.
        """
        embed = self.embed(document)
        self.documents[doc_idx] = document
        self.vectors[doc_idx] = embed

    def persist(self) -> None:
        """Save the vector database to disk.

        Raises
        ------
        AttributeError
            If db_path is None, indicating no valid file location was provided.

        Notes
        -----
        Uses pickle to serialize both vectors and documents to the file specified
        by db_path. The entire database state is saved and can be restored using
        the from_path class method.
        """
        if self.db_path is None:
            raise AttributeError("Cannot persist DB if db_path is a valid file location")
        import pickle
        data = pickle.dumps([self.vectors, self.documents])
        with open(self.db_path, 'wb') as f:
            f.write(data)

    @classmethod
    def from_path(cls, path: str) -> SimpleVectorDB:
        """Load a vector database from a saved file.

        Parameters
        ----------
        path : str
            Path to the saved database file.

        Returns
        -------
        SimpleVectorDB
            A new SimpleVectorDB instance loaded from the saved file.

        Notes
        -----
        Loads a database that was previously saved using the persist method.
        The loaded database will have the same db_path as the file it was loaded from.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        pickle.UnpicklingError
            If the file is corrupted or not a valid pickle file.
        """
        with open(path, 'rb') as f:
            import pickle
            vectors, documents = pickle.loads(f.read())
        return cls(vectors, documents, path)

    def _document_query_releveance(self, embedding: np.ndarray):
        """Calculate similarity scores between query embedding and all document vectors.

        This is a private method used internally for similarity computation.
        """
        return self.vectors @ embedding


def k_largest_idx(x: Sequence, k: int):
    """Find indices of k largest elements in an array-like sequence.

    Parameters
    ----------
    x : array_like
        Input array from which to find the largest elements.
    k : int
        Number of largest elements to find.

    Returns
    -------
    np.ndarray
        Array of indices corresponding to the k largest elements in x.
        The indices are not sorted by value - they appear in arbitrary order.

    Notes
    -----
    Uses numpy's argpartition for efficient O(n) partitioning rather than
    full sorting. This is more efficient than argsort when k << len(x).

    Examples
    --------
    >>> x = [1, 5, 3, 9, 2]
    >>> k_largest_idx(x, 2)
    array([3, 1])  # indices of values 9 and 5
    """
    x_arr = np.asarray(x)
    partition = np.argpartition(x_arr, x_arr.size - k)
    return partition[-k:]
