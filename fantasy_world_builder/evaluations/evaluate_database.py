import json
import os
import numpy as np
from fantasy_world_builder.database import SimpleVectorDB
from fantasy_world_builder.builder import World
from fantasy_world_builder.database import EmbeddingsStore, k_largest_sorted_idx
from fantasy_world_builder.serialize import PrettyRoundTripJSONEncoder, pretty_roundtrip_decoder

from evaluations.metrics import mean_precision_at_k

def load_world_from_experiments(world_name):
    rel_path = '/home/philosofool/repos/philosofool/local_data/worlds/world_15.json'
    path = os.path.join(rel_path, f"{world_name}.json")
    with open(path, 'r') as f:
        world = json.loads(f.read(), object_hook=pretty_roundtrip_decoder)
    return world

def load_world(world_name):
    world = load_world_from_experiments(world_name)['world']
    return World(**world)


def database():
    db_path = '/home/philosofool/repos/philosofool/local_data/worlds/vielbound.vdb'
    try:
        database = SimpleVectorDB.from_path(db_path)
    except FileNotFoundError:
        database = SimpleVectorDB(np.array([]), [], db_path)
        world = load_world('world_15')
        for i, v in enumerate(world.entities.values()):
            as_string = json.dumps(v, cls=PrettyRoundTripJSONEncoder(max_inline_list=25))
            print(as_string[:50])
            database.add_document(as_string)
        database.persist()
    return database

def embedding_store():
    emb_path = '/home/philosofool/repos/philosofool/local_data/worlds/evaluation_embeddings.json'
    try:
        store = EmbeddingsStore.from_path(emb_path)
    except FileNotFoundError:
        store = EmbeddingsStore({}, emb_path)
    return store


def queries():
    return [
        ("The world of Vielbound", [0]),
        ("A fantasy world that would be suited as the setting of a Dungeons and Dragons game.", [0]),
        ("Where you would find pixies?", [1]),
        ("The souls of living dwell here, after life.", [2]),
        ("The most infernal realm", [3]),
        ("A place of violent chaos.", [4]),
        ("Where does magic come from?", [5]),
        ("Mines abound in this area.", [6]),
        ("It's a culturally diverse place, owing to the trade which passes through.", [7]),
        ("Mysterious forests inspire fantasies in travelers, which leads them here.", [8]),
        ("A hard frontier world, inhabited only by those who have devised many means of dealing with arid, hot and harsh climes.", [9]),
        ("You won't find enough fuel for fires here, though you will wish you could.", [10]),
        ("A place of farmers and shepherds. Pastoral in the best regions, but elsewhere hot and dry.", [11]),
    ]


def load_results() -> list:
    path = '/home/philosofool/repos/philosofool/local_data/worlds/evaluation_results.json'
    try:
        with open(path, 'r') as f:
            results = json.loads(f.read(), object_hook=pretty_roundtrip_decoder)
    except FileNotFoundError:
        return []
    return results

def save_results(results: list) -> None:
    path = '/home/philosofool/repos/philosofool/local_data/worlds/evaluation_results.json'
    with open(path, 'w') as f:
        f.write(json.dumps(results, cls=PrettyRoundTripJSONEncoder, indent=2))


class TestEmbedding:
    def __init__(self, database: SimpleVectorDB, store: EmbeddingsStore):
        self.database = database
        self.store = store

    def test(self, query: str, expected: list) -> dict:
        results = self._score(query, expected)
        keys = ['query', 'expected', 'retrieved', 'avg_precision_at_k']
        return {k: v for k, v in zip(keys, results)}

    def _score(self, query: str, expected: list) -> list:
        try:
            embedding = self.store.get(query)
        except KeyError:
            embedding = self.database.embed(query)
            self.store.add(query, embedding)
        scores = self.database._document_query_relevance(embedding)
        doc_indexes = k_largest_sorted_idx(scores, len(expected) + 2).tolist()
        ap_at_k = mean_precision_at_k(doc_indexes, expected)
        return [query, expected, doc_indexes, ap_at_k]

    @classmethod
    def default(cls):
        store = embedding_store()
        vector_db = database()
        return cls(vector_db, store)


def execute_tests():
    results = []
    tests = queries()
    test_embeddings = TestEmbedding.default()
    for query, expected in tests:
        result = test_embeddings.test(query, expected)
        results.append(result)
    test_embeddings.store.save()
    save_results(results)


if __name__ == '__main__':
    execute_tests()
