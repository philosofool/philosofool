import sys


# sys.path.append('/home/philosofool/repos/philosofool/fantasy_world_builder')

import numpy as np
import pytest
from fantasy_world_builder.database import k_largest_idx, SimpleVectorDB



def test_k_largest_idx():
    arr = [4, 1, 6, 2, 10, 6, 3]
    expected = [2, 4, 5]
    assert k_largest_idx(arr, 1) == [4]
    assert len(np.intersect1d(k_largest_idx(arr, 3), expected)) == 3

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
    vector_db = SimpleVectorDB(np.array([[1., 0], [0, 2], [0, 1]]), ['dummy', 'dummy2'], None)
    result = vector_db._document_query_releveance(np.array([0, 1]))
    np.testing.assert_array_almost_equal(result, np.array([0, 2, 1]))

def test_simple_db__embed(vector_db: SimpleVectorDB):
    embedding = vector_db.embed('This is a test sentence')
    assert embedding.shape == (vector_db._embed_dim,)

def test_simple_db__get_documents(vector_db):
    result = vector_db.get_documents("Someone answers a trivia question.", n_results=1)
    assert 'Norm' in result[0]
