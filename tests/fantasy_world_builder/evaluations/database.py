import json
import os
import numpy as np
from fantasy_world_builder.database import SimpleVectorDB
from fantasy_world_builder.builder import World

def load_world_from_experiments(world_name):
    rel_path = '/home/philosofool/repos/philosofool/fantasy_world_builder/worlds'
    path = os.path.join(rel_path, f"{world_name}.json")
    with open(path, 'r') as f:
        world = json.loads(f.read())
    return world

def load_world(world_name):
    world = load_world_from_experiments(world_name)['world']
    return World(**world)


def database():
    db_path = '/home/philosofool/repos/philosofool/fantasy_world_builder/worlds/vielbound.vdb'
    try:
        database = SimpleVectorDB.from_path(db_path)
    except FileNotFoundError:
        database = SimpleVectorDB(np.array([]), [], db_path)
        world = load_world('world_15')
        for i, v in enumerate(world.entities.values()):
            as_string = json.dumps(v)
            print(as_string[:50])
            database.add_document(as_string)
        database.persist()
    return database

def queries():
    return [
        ("The world of Vielbound", 0),
        ("A fantasy world that would be suited as the setting of a Dungeons and Dragons game.", 0),
        ("Where you would find pixies?", 1),
        ("The souls of living dwell here, after life.", 2),
        ("The most infernal realm", 3),
        ("A place of violent chaos.", 4),
        ("Where does magic come from?", 5),
        ("Mines abound in this realm.", 6),
        ("It's a culturally diverse place, owing to the trade which passes through.", 7),
        ("Mysterious forests inspire fantasies in travelers, which leads them here.", 8),
        ("A hard frontier world, inhabited only by those who have devised many means of dealing with arid, hot and harsh climes.", 9),
        ("You won't find enough fuel for fires here, though you will wish you could.", 10),
        ("A place of farmers and shepherds. Pastoral in the best regions, but elsewhere hot and dry.", 11)
    ]


def load_results() -> list:
    path = '/home/philosofool/repos/philosofool/fantasy_world_builder/worlds/evaluation_results.json'
    try:
        with open(path, 'r') as f:
            results = json.loads(f.read())
    except FileNotFoundError:
        return []
    return results

def save_results(results: list) -> None:
    path = '/home/philosofool/repos/philosofool/fantasy_world_builder/worlds/evaluation_results.json'
    with open(path, 'w') as f:
        f.write(json.dumps(results))


def test_db(database: SimpleVectorDB, query: str, expected):
    doc_indexes = database._lookup_idx(query, n_results=3)
    return [query, expected, doc_indexes, doc_indexes[0] == expected, expected in doc_indexes]

def execute_tests():
    existing_results = load_results()
    tests = queries()
    vector_db = database()
    if len(tests) > len(existing_results):
        for query, expected in tests[len(tests):]:
            result = test_db(vector_db, query, expected)
            existing_results.append(result)
    save_results(existing_results)


if __name__ == '__main__':
    execute_tests()
