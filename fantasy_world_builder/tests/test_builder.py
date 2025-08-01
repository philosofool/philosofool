import json
from typing import Dict
from langchain_core.runnables import Runnable
from pydantic import BaseModel
import pytest

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from fantasy_world_builder.builder import Builder, World
from fantasy_world_builder.schema import Entity, Setting, Character



@pytest.fixture
def world() -> World:
    return World('A town in the mountains.', {}, {})


@pytest.fixture
def entity() -> Entity:
    return {'name': "Chris", 'summary': "An all-around swell person."}

def test_world_basic(world):
    assert world.graph == {}
    assert world.entities == {}
    assert world.description == 'A town in the mountains.'

def test_world_add_entity(world, entity):
    world.add_entity(entity)
    assert len(world.graph) == 1
    first_key = next(iter(world.graph))
    edges = world.graph[first_key]
    assert type(first_key) is str
    assert type(edges) is set

def test_world_add_entity_updates_relationships(world, entity):
    world.add_entity(entity)
    entity2 = {'name': 'Rachel', 'description': "Chris's friend."}
    world.add_entity(entity2, ["Chris"])
    chris_edges = world.graph["Chris"]
    rachel_edges = world.graph["Rachel"]
    assert 'Chris' in rachel_edges
    assert 'Rachel' in chris_edges

def test_builder(world):
    builder = Builder(world, FakeListChatModel(responses=['Hello!']))
    assert isinstance(builder.world, World)
    assert isinstance(builder.llm, BaseChatModel)

def test_builder_create_entity(world):
    test_character = {
        "name": "Tom",
        "summary": "A curious cat who often gets into trouble.",
        "background": "Tom has lived in the countryside since he was a kitten, constantly chasing after his rival, Jerry.",
        "appearance": "Grey fur, white paws, and expressive green eyes.",
        "personality": "Impulsive, determined, and often outsmarted.",
        "details": "Recently caught a mouse, but it turned out to be a toy."
    }

    class FakeWithStructure(FakeListChatModel):
        def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
            return self
            #r# eturn super().with_structured_output(schema, include_raw=include_raw, **kwargs)

    llm = FakeWithStructure(responses=['CHARACTER', json.dumps(test_character)])
    builder = Builder(world, llm)
    result_character = builder.create_entity("A cat.")
    assert result_character == test_character

def test_builder_add_entity(world):
    test_character = {
        "name": "Tom",
        "summary": "A curious cat who often gets into trouble.",
        "background": "Tom has lived in the countryside since he was a kitten, constantly chasing after his rival, Jerry.",
        "appearance": "Grey fur, white paws, and expressive green eyes.",
        "personality": "Impulsive, determined, and often outsmarted.",
        "details": "Recently caught a mouse, but it turned out to be a toy."
    }

    class FakeWithStructure(FakeListChatModel):
        def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
            return self
            #r# eturn super().with_structured_output(schema, include_raw=include_raw, **kwargs)

    llm = FakeWithStructure(responses=['CHARACTER', json.dumps(test_character)])
    builder = Builder(world, llm)
    builder.add_entity("A cat.")
    assert builder.world.entities['Tom'] == test_character


@pytest.mark.parametrize('prompt, llm_responses, expected_schema', [
    ("Create a setting.", ['SETTING'], Setting),
    ('Create a person.', ['CHARACTER'], Character),
    ('Create a thing.', ['OTHER'], Entity),
    ('Create a largest prime number.', ['halucination'], Entity)
])
def test_builder__engineer_prompt(world, prompt, llm_responses, expected_schema):
    builder = Builder(world, FakeListChatModel(responses=llm_responses))

    schema, engineered_prompt = builder._engineer_prompt(prompt)
    assert schema == expected_schema
    assert prompt in engineered_prompt
    assert schema.__name__.upper() in engineered_prompt

def test_fake_chat_model():
    # assure this entity behaves like I think.
    fake_chat = FakeListChatModel(responses=['Hello', 'Goodbye'])
    result = fake_chat.invoke([HumanMessage('Hello')])
    assert result.content == 'Hello'
    assert fake_chat.invoke('Bye.').content == 'Goodbye'
    assert fake_chat.invoke('Hello again.').content == 'Hello'

def test_builder_retrieve_setting_info(world):
    llm = init_chat_model('openai:gpt-4.1-nano') # .bind_tools([retrieve_setting_details])
    builder = Builder(world, llm)
    retrieve_setting_details = builder.retrieve_setting_details

    tool_input_schema = retrieve_setting_details.get_input_schema().model_json_schema()
    # assert 'setting_entities' in tool_input_schema['properties']
    tool_call_schema = retrieve_setting_details.tool_call_schema.model_json_schema()
    # assert 'setting_entities' not in tool_call_schema['properties']
    assert 'details' in tool_call_schema['properties']

    llm_with_tools = llm.bind_tools([retrieve_setting_details])
    ai_message = llm_with_tools.invoke('Create a character who knows Jerry and Tom')
    tool_calls = ai_message.tool_calls
    assert len(tool_calls) == 1
    args = tool_calls[0]['args']['details']
    assert 'Jerry' in args
    assert 'Tom' in args
    assert len(args) == 2
