import json
import textwrap
from collections.abc import Callable, Mapping, Iterable, Sequence, Generator
from typing import Any, Type, Annotated

from fantasy_world_builder.schema import Setting, Character, Entity
from random import random

from langchain_core.tools import InjectedToolArg, tool
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel



class World:
    """A Model of a World, including setting and characters."""
    def __init__(self, description: str, graph: dict[str, set], entities: dict[str, Entity]):
        self.description = description
        self.graph = graph
        self.entities = entities

    def add_entity(self, entity: Entity, relationships: Iterable = tuple()):
        key = entity['name']
        if key in self.graph:
            raise KeyError("Entity is already in the world.")
        relationships_set = set()
        for name in relationships:
            if name not in self.graph:
                raise KeyError("All relationships must be to known entities.")
            relationships_set.add(name)
            self.graph[name].add(key)
        self.graph[key] = relationships_set
        self.entities[key] = entity


class Builder:
    def __init__(self, world: World, llm: BaseChatModel):
        self.world = world
        self.llm = llm

    def create_entity(self, prompt: str | None) -> Entity:
        schema, embedded_prompt = self._engineer_prompt(prompt)

        llm = self.llm.with_structured_output(schema)

        response = llm.invoke(embedded_prompt).content
        response_dict = json.loads(response)
        return response_dict

    def add_entity(self, prompt: str, relationships: Iterable[str] = tuple()):
        if relationships:
            raise NotImplementedError("Relationship creation is not yet implemented.")
        entity = self.create_entity(prompt)
        self.world.add_entity(entity)

    @tool
    def retrieve_setting_details(self, details: list[str]) -> list[Entity]:
        """Get info on several named entities in a setting, returning that information.

        details:
            A list of names, each representing an entity whose setting information will be retrieved.
        setting_entities:
            A dictionary of names, such as found in details, and dictionaries of information about them.
        """
        setting_entities = self.world.entities
        retrieved_details = []
        for detail in details:
            entity_detail = setting_entities.get(detail)
            if entity_detail is not None:
                retrieved_details.append(entity_detail)
        return retrieved_details

    def _engineer_prompt(self, prompt: str) -> tuple[Type[Entity], str]:
        response = self.llm.invoke(f"Is this prompt\n    {prompt}\n looking for a character or setting or something else? Answer CHARACTER or SETTING or OTHER.").content
        world_description = self.world.description
        entity_name = response
        if response.upper() == 'SETTING':
            schema = Setting
        elif response.upper() == 'CHARACTER':
            schema = Character
        else:
            entity_name = f'ENTITY besides a person or place'.upper()
            schema = Entity

        engineered_prompt = textwrap.dedent(
            f""""
            Create a {entity_name} within {world_description}.
            Here's a basic concept to get started on this {entity_name}: {prompt}
            """)
        return schema, engineered_prompt
