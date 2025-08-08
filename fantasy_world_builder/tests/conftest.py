from fantasy_world_builder.database import SimpleVectorDB
from langchain.chat_models import init_chat_model
import numpy as np
import pytest

@pytest.fixture
def vector_db() -> SimpleVectorDB:
    documents = ['Sam is a bartender', 'Norm is a patron who knows lots of trivia']
    path = 'test_simple_db.vdb'
    try:
        db = SimpleVectorDB.from_path(path)
    except FileNotFoundError:
        db = SimpleVectorDB(np.array([]), [], path)
        for doc in documents:
            db.add_document(doc)
        db.persist()
    return db

@pytest.fixture(scope='session')
def llm():
    return init_chat_model('openai:gpt-4.1-nano')