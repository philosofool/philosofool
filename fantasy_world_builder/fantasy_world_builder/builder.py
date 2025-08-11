from collections.abc import Iterable
import json
import itertools
import textwrap
from typing import Type, Annotated, TypedDict, Any

from fantasy_world_builder.writer import WriterState
from fantasy_world_builder.schema import (  # noqa: F401  T
    Setting, Character, Entity, List,
    character_schema, setting_schema, entity_schema
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage  # noqa: F401
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.runnables import Runnable


# TODO: move these to a better location (serialize.py?)
def serialize_sets(obj: Any) -> Any:
    if isinstance(obj, set):
        return {"__type__": "set", "items": list(obj)}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def deserialize_sets(obj: dict) -> Any:
    if "__type__" in obj and obj["__type__"] == "set":
        return set(obj["items"])
    return obj

class World:
    """A Model of a World, including setting and characters.

    Stores entity data and their relationships in a graph structure.
    """
    def __init__(self, description: Entity, graph: dict[str, set], entities: dict[str, Entity]):
        self.description = description
        self.graph = graph
        self.entities = entities

    def add_entity(self, entity: Entity, relationships: Iterable[str] = tuple()):
        """Add a new entity to the world and connect it to existing entities.

        Args:
            entity: The entity to add.
            relationships: Other entities this one is related to.

        Raises:
            ValueError: If the entity already exists.
            KeyError: If any relationships refer to unknown entities.
        """
        key = entity['name']
        if key in self.graph:
            raise ValueError("Entity is already in the world.")
        relationships_set = set()
        for entity_ in relationships:
            name = entity_
            if name not in self.graph:
                raise KeyError("All relationships must be to known entities.")
            relationships_set.add(name)
            self.graph[name].add(key)
        self.graph[key] = relationships_set
        self.entities[key] = entity

    def save(self, path):
        as_dict = {key: getattr(self, key) for key in ['graph', 'entities', 'description']}
        to_string = json.dumps(as_dict, default=serialize_sets)
        with open(path, 'w') as f:
            f.write(to_string)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            as_dict = json.loads(f.read(), object_hook=deserialize_sets)
        return cls(**as_dict)

def create_build_world(world: World):
    def build(state: WriterState):
        entity = json.loads(state['messages'][-1].content)
        world.add_entity(entity)
        return {'messages': [SystemMessage(f"Successfully added {entity.get('topic', entity['name'])} to world.")]}
    return build
