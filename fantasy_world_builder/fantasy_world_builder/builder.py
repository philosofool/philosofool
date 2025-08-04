from collections.abc import Iterable
import json
import itertools
from pprint import pprint
import textwrap
from typing import Type, Annotated, TypedDict

from fantasy_world_builder.llm import node_factory
from fantasy_world_builder.schema import (
    Setting, Character, Entity, List,
    character_schema, setting_schema, entity_schema
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage  # noqa: F401
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.runnables import Runnable

class World:
    """A Model of a World, including setting and characters."""
    def __init__(self, description: Entity, graph: dict[str, set], entities: dict[str, Entity]):
        self.description = description
        self.graph = graph
        self.entities = entities

    def add_entity(self, entity: Entity, relationships: Iterable[Entity] = tuple()):
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

class BuildState(TypedDict):
    messages: Annotated[list, add_messages]
    research: Annotated[list[Entity], 'The results of the most recent research completed.']
    schema: Annotated[Type[Entity], "The schema type to use when writing."]

class SupervisorNode:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm  # NOTE: this attribute is currently unsused.

    def compile(self) -> Runnable:
        results = itertools.cycle(['RESEARCH', 'WRITE', 'BUILD', 'FINISHED'])

        def supervisor_node(state: BuildState):
            response_text = next(results)
            return {'messages': [AIMessage(response_text)]}

        return supervisor_node    # pyright: ignore [reportReturnType]


class ResearchNode:
    def __init__(self, world: World, llm: BaseChatModel):
        self.llm = llm
        self.world = world

    def build_graph(self) -> StateGraph:

        def search_node(state: BuildState) -> dict:
            prompt = state['messages'][-2]
            llm = self.llm.with_structured_output(List)
            search_prompt = f""""What are the things named in "{prompt}"""""
            ai_message = llm.invoke(search_prompt)
            entities = ai_message['items']    # pyright: ignore [reportIndexIssue]
            setting_info = [self.world.description] + [self.world.entities[entity] for entity in entities if entity in self.world.entities]
            return {'research': setting_info}

        graph = StateGraph(BuildState)
        graph.add_node(search_node)
        graph.add_edge(START, 'search_node')
        graph.add_edge('search_node', END)
        return graph

    def compile(self) -> Runnable:
        graph = self.build_graph()
        return graph.compile()

class WriterNode:
    def __init__(self, role: str, llm: BaseChatModel, memory):
        self.role = role
        self.llm = llm
        self.memory = memory

    def compile(self) -> Runnable:
        def prompt_context(state: BuildState):
            prompt = state['messages'][0].content
            response: str = (
                self.llm.invoke(
                    f"Is this prompt\n    {prompt}\n looking for a character or setting or something else? Answer CHARACTER or SETTING or OTHER."
                ).content    # pyright: ignore [reportAssignmentType]
            )
            entity_name = response

            if response.upper() == 'SETTING':
                schema = setting_schema
            elif response.upper() == 'CHARACTER':
                schema = character_schema
            else:
                entity_name = 'ENTITY besides a person or place'.upper()
                schema = entity_schema

            engineered_prompt = textwrap.dedent(
                f""""
                Create a {entity_name} within this setting.
                Here's a basic concept to get started on this {entity_name}: {prompt}
                """)
            return {'messages': [SystemMessage(self.role), AIMessage(engineered_prompt)], 'schema': schema}

        def write_node(state: BuildState, config: dict):
            research = state['research']
            messages = state['messages'] + [SystemMessage(json.dumps(research))]
            llm = self.llm.with_structured_output(state['schema'])
            response = llm.invoke(messages, config)

            as_message = AIMessage(json.dumps(dict(response)))
            return {'messages': [as_message]}

        graph = StateGraph(BuildState)
        graph.add_node(prompt_context)
        graph.add_node(write_node)
        graph.add_edge(START, 'prompt_context')
        graph.add_edge('prompt_context', 'write_node')
        graph.add_edge('write_node', END)
        return graph.compile(self.memory)

class BuildWorldNode:
    def __init__(self, world: World):
        self.world = world

    def compile(self) -> Runnable:
        def add_entity_to_world(state: BuildState):
            entity = json.loads(state['messages'][-2].content)
            research = state['research'][1:]
            relationships = [entity['name'] for entity in research]
            self.world.add_entity(entity, relationships=relationships)

        graph = StateGraph(BuildState)
        graph.add_node(add_entity_to_world)
        graph.add_edge(START, 'add_entity_to_world')
        graph.add_edge('add_entity_to_world', END)
        return graph.compile()


class SettingCreator:
    def __init__(self, supervisor: SupervisorNode, researcher: ResearchNode, writer: WriterNode, builder: BuildWorldNode):
        self.supervisor = supervisor.compile()
        self.researcher = researcher.compile()
        self.writer = writer.compile()
        self.builder = builder.compile()

    @staticmethod
    def routing(state: BuildState):
        message = state['messages'][-1].content

        if message == 'RESEARCH':
            return 'researcher'
        if message == 'WRITE':
            return 'writer'
        if message == 'BUILD':
            return 'builder'
        if message == 'FINISHED':
            return END
        raise ValueError(f"Unexpected routing message '{message}' received.")

    def build_graph(self) -> StateGraph:
        graph = StateGraph(BuildState)
        graph.add_node('supervisor', self.supervisor)
        graph.add_node('researcher', self.researcher,)
        graph.add_node('writer', self.writer)
        graph.add_node('builder', self.builder)

        graph.add_edge(START, 'supervisor')
        graph.add_conditional_edges('supervisor', self.routing)
        graph.add_edge('researcher', 'supervisor')
        graph.add_edge('writer', 'supervisor')
        graph.add_edge('builder', 'supervisor')
        return graph

    def compile(self, checkpointer: MemorySaver | None) -> Runnable:
        graph = self.build_graph()
        return graph.compile(checkpointer)

    @classmethod
    def from_llm_memory(cls, llm, memory, world: World, **kwargs):

        supervisor = kwargs.get('supervisor', SupervisorNode(llm))
        researcher = kwargs.get('researcher', ResearchNode(world, llm))
        role = f"Write about {world.description['name']} which is {world.description['summary']}"
        writer = kwargs.get('writer', WriterNode(role, llm, memory))
        builder = kwargs.get('builder', BuildWorldNode(world))
        return cls(supervisor, researcher, writer, builder)
