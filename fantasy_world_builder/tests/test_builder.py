import json
from typing import Dict
from langchain_core.runnables import Runnable
from pydantic import BaseModel
import pytest

import numpy as np

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from fantasy_world_builder.builder import  World
from fantasy_world_builder.builder import BuildWorldNode, WriterNode, ResearchNode, SupervisorNode, SettingCreator
from fantasy_world_builder.schema import Entity, Setting, Character, List, character_schema
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END



@pytest.fixture(scope='module')
def llm():
    return init_chat_model('openai:gpt-4.1-nano')

@pytest.fixture
def world() -> World:
    return World({'name': 'Mountain Town', 'summary': 'A town in the mountains.'}, {}, {})


@pytest.fixture
def entity() -> Entity:
    return {'name': "Chris", 'summary': "An all-around swell person."}

def test_world_basic(world):
    assert world.graph == {}
    assert world.entities == {}
    assert world.description['summary'] == 'A town in the mountains.'

def test_world_add_entity(world, entity):
    world.add_entity(entity)
    assert len(world.graph) == 1
    first_key = next(iter(world.graph))
    edges = world.graph[first_key]
    assert type(first_key) is str
    assert type(edges) is set

def test_world_add_entity_raises(world, entity):
    world.add_entity(entity)
    with np.testing.assert_raises(ValueError):
        world.add_entity(entity)

def test_world_add_entity_updates_relationships(world, entity):
    world.add_entity(entity)
    entity2 = {'name': 'Rachel', 'summary': "Chris's friend."}
    world.add_entity(entity2, [entity['name']])
    chris_edges = world.graph["Chris"]
    rachel_edges = world.graph["Rachel"]
    assert 'Chris' in rachel_edges
    assert 'Rachel' in chris_edges

def test_world_save_load(world, entity):
    from tempfile import TemporaryDirectory
    import os
    temp_dir = TemporaryDirectory()
    directory = temp_dir.name
    world.add_entity(entity)
    path = os.path.join(directory, 'world.json')
    world.save(path)
    assert os.path.exists(path)
    round_trip = World.load(path)
    assert world.graph == round_trip.graph
    assert world.entities == round_trip.entities
    assert world.description == round_trip.description

def test_WriterNode(llm, world, entity):
    writer = WriterNode(
        'Write about a mountain town.',
        llm=llm,
        memory=MemorySaver()
    ).compile()
    state = {
        'messages': [{'role': 'user', 'content': 'Tell me about Chris.'}],
        # 'research': [world.description, entity]
        'research': []

    }
    config = {"configurable": {"thread_id": "1"}}
    response = writer.invoke(state, config=config)
    assert response['schema'] == character_schema
    chris = json.loads(response['messages'][-1].content)
    assert 'personality' in chris, 'The response final message should be a Character type dict.'

def test_WriteNode_memory(llm, world, entity):
    writer = WriterNode(
        'You create settings in about a mountain town.',
        llm=llm,
        memory=MemorySaver()
    ).compile()
    state = {
        'messages': [{'role': 'user', 'content': 'Create village called Hamlet in a fictional world.'}],
        'research': []
    }
    config = {"configurable": {"thread_id": "1"}}
    response = writer.invoke(state, config=config)
    hamlet = json.loads(response['messages'][-1].content)
    assert hamlet['name'] == 'Hamlet'
    state = {
        'messages': [{'role': 'user', 'content': 'A person in Hamlet.'}],
        'research': [world.description, hamlet]
        # 'research': []
    }
    response = writer.invoke(state, config=config)
    person = json.loads(response['messages'][-1].content)
    assert person['name'] != 'Hamlet'


def test_WorldBuildNode(world, entity):
    builder = BuildWorldNode(world).compile()
    world.add_entity(entity)
    rachel = {'name': 'Rachel', 'summary': "Chris's friend."}
    state = {
        'messages': [AIMessage(json.dumps(rachel)), 'placeholder for AI message "BUILD"'],
        'research': [..., entity]
    }
    response = builder.invoke(state)
    assert 'Chris' in world.graph
    assert 'Rachel' in world.graph
    assert world.entities['Chris'] == entity

def test_SettingCreator(world, llm):
    memory = MemorySaver()
    setting_creator = SettingCreator.from_llm_memory(llm, memory, world).compile(memory)
    state = {'messages': [HumanMessage('Make a cat named Tom')]}
    config = {'configurable': {'thread_id': '1'}}
    final_state = setting_creator.invoke(state, config)
    assert 'research' in final_state
    assert 'schema' in final_state

@pytest.mark.parametrize('input, expected', [
    ('RESEARCH', 'researcher'),
    ('WRITE', 'writer'),
    ('BUILD', 'builder'),
    ('FINISHED', END)
])
def test_SettingCreator_routing(input, expected):
    state = {'messages': [AIMessage(input)]}
    assert SettingCreator.routing(state) == expected


def test_fake_chat_model():
    # assure this entity behaves like I think.
    fake_chat = FakeListChatModel(responses=['Hello', 'Goodbye'])
    result = fake_chat.invoke([HumanMessage('Hello')])
    assert result.content == 'Hello'
    assert fake_chat.invoke('Bye.').content == 'Goodbye'
    assert fake_chat.invoke('Hello again.').content == 'Hello'

class TestLLMCalls:

    @pytest.fixture(scope='class')
    def llm(self) -> BaseChatModel:
        return init_chat_model('openai:gpt-4.1-nano')

    @pytest.mark.parametrize('prompt, expected', [
        ('What are the things named in "Tom and Jerry are friends"', {"Tom", "Jerry"}),
        ('What are the things named in "Luke, Darth, and Ben went for a picnic on Tatooine."', {'Luke', 'Darth', 'Ben', 'Tatooine'}),
        ("""What are the things named in "A character who knows Alex"?""", {"Alex"})

    ])
    def test_list_structured_calls(self, prompt, expected, llm):
        llm_with_list = llm.with_structured_output(List)
        result = llm_with_list.invoke(prompt)
        assert 'items' in result
        assert expected.union(result['items']) == expected
        assert expected.difference(result['items']) == set()

    @pytest.mark.parametrize('prompt, condition', [
        ('Write three sentences about mountain biking.', lambda x: len(x) == 3),
    ])
    def test_list_structured_calls(self, prompt, condition, llm):
        llm_with_list = llm.with_structured_output(List)
        result = llm_with_list.invoke(prompt)
        assert 'items' in result
        assert condition(result['items'])


def test_composite_graph_memory():
    from typing import Annotated, TypedDict
    from langgraph.graph import StateGraph, add_messages, START, END
    from langchain_core.messages import HumanMessage

    llm = init_chat_model('openai:gpt-4.1-nano')

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def subgraph_node(state):
        return {'messages': [llm.invoke(state['messages'])]}

    subgraph = StateGraph(State)
    subgraph.add_node(subgraph_node)
    subgraph.add_edge(START, 'subgraph_node')
    subgraph.add_edge('subgraph_node', END)
    component = subgraph.compile(MemorySaver())
    config = {'configurable': {'thread_id': '1'}}

    # Assure that subgraph memory works as expected.
    component.invoke({'messages': [HumanMessage('My name is Ruben.')]}, config)
    response = component.invoke({'messages': [HumanMessage("What's my name?")]}, config)
    latest_message = response['messages'][-1].content
    assert 'Ruben' in latest_message, f"Memory is expected between calls."

    graph = StateGraph(State)
    graph.add_node('subgraph', component)
    graph.add_edge(START, 'subgraph')
    graph.add_edge('subgraph', END)
    app = graph.compile(MemorySaver())

    # Assure that full-graph memory works as expected.
    app.invoke({'messages': [HumanMessage('My name is Ruben.')]}, config)
    response = app.invoke({'messages': [HumanMessage("What's my name?")]}, config)
    latest_message = response['messages'][-1].content
    assert 'Ruben' in latest_message, f"Memory is expected between calls. {latest_message}"
