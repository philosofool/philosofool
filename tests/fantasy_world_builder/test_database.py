from tempfile import TemporaryDirectory

import os
import pytest
import numpy as np
from fantasy_world_builder.database import k_largest_idx, k_largest_sorted_idx, SimpleVectorDB

def test_k_largest_idx():
    arr = [4, 1, 6, 2, 10, 6, 3]
    expected = [2, 4, 5]
    assert k_largest_idx(arr, 1) == [4]
    assert len(np.intersect1d(k_largest_idx(arr, 3), expected)) == 3

def test_k_largest_sorted_idx():
    arr = [4, 1, 6, 2, 10, 6, 3]
    expected = [4, 5, 2]
    np.testing.assert_array_equal(k_largest_sorted_idx(arr, 3), expected)

def test_simple_db__init():
    documents = ['Sam is a bartender', 'Norm is a patron who knows lots of trivia']
    path = 'test_simple_db.vdb'
    vectors = np.array([[1., 1.1, -1], [-4, .05, .01]])
    database = SimpleVectorDB(vectors, documents, path)
    assert np.array_equal(database.vectors, vectors)
    assert database.documents == documents
    assert database.db_path == path
    assert database._embed_dim == 3, 'This is an implementation detail, but current implementation expects this.'

def test__document_query_relevance():
    vector_db = SimpleVectorDB(np.array([[1., 0], [0, 2], [0, 1]]), ['dummy', 'dummy2'])
    result = vector_db._document_query_releveance(np.array([0, 1]))
    np.testing.assert_array_almost_equal(result, np.array([0, 2, 1]))

def test_simple_db__embed(vector_db: SimpleVectorDB):
    embedding = vector_db.embed('This is a test sentence')
    assert embedding.shape == (vector_db._embed_dim,)

def test_simple_db__get_documents(vector_db):
    result = vector_db.get_documents("Someone answers a trivia question.", n_results=1)
    assert 'Norm' in result[0]
    result = vector_db.get_documents("Someone answers a trivia question.", n_results=2)
    assert 'Norm' in result[0], "Results should be sorted, descending."

def test_simple_db__get_documents_too_many(vector_db):
    result = vector_db.get_documents("Someone answers a trivia question.", n_results=5)
    assert len(result) == len(vector_db.documents), "When requesting more documents than are known, return them all."


def test_simple_db__persist1():
    temp_dir = TemporaryDirectory()
    path = os.path.join(temp_dir.name, 'db.vdb')
    vector_db = SimpleVectorDB(np.array([[1., 0], [0, 2], [0, 1]]), ['dummy', 'dummy2'], None)
    with np.testing.assert_raises(AttributeError):
        vector_db.persist()
    vector_db.persist(path)
    assert os.path.exists(path)

def test_simple_db__persist2():
    temp_dir = TemporaryDirectory()
    path = os.path.join(temp_dir.name, 'db.vdb')
    vector_db = SimpleVectorDB(np.array([[1., 0], [0, 2], [0, 1]]), ['dummy', 'dummy2'], path)
    vector_db.persist()
    assert os.path.exists(path)

def test_simple_db__from_path():
    temp_dir = TemporaryDirectory()
    path = os.path.join(temp_dir.name, 'db.vdb')
    vector_db = SimpleVectorDB(np.array([[1., 0], [0, 2], [0, 1]]), ['dummy', 'dummy2'], path)
    vector_db.persist()
    round_trip = SimpleVectorDB.from_path(path)
    np.testing.assert_array_equal(round_trip.vectors, vector_db.vectors)
    assert round_trip.documents == vector_db.documents
