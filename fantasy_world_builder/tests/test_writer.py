import json
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from fantasy_world_builder.writer import (
    create_writer_node, create_character_node, create_setting_node, WriterState,
    create_detail_node, writer_graph
)
import pytest

def human_message(message) -> WriterState:
    return {'messages': [HumanMessage(message)], 'routing': 'writer'}

def to_compiled_graph(node, memory=None):
    graph = StateGraph(WriterState)
    graph.add_node('node', node)
    graph.set_entry_point('node')
    graph.set_finish_point('node')
    return graph.compile(memory)

def _are_similar_responses(llm, response1, response2, condition) -> str:
    prompt = SystemMessage(condition)
    return llm.invoke([prompt, response1, response2]).content


@pytest.mark.parametrize('message_str, expected', [
    ('A person who likes dogs.', 'CHARACTER'),
    ('A place for happy dogs to run around.', 'SETTING'),
    ('Things happy dogs like.', 'DETAIL'),
    ('A historian who writes detailed books', 'CHARACTER')
])
def test_create_writer_node(message_str, expected, llm):
    writer_node = create_writer_node(llm)
    message = human_message(message_str)
    response = writer_node(message)
    assert 'routing' in response
    assert response['routing'].content == expected


def test_create_character_node(llm):
    create_character = create_character_node(llm)
    response: dict = create_character(human_message('A person who likes dogs.'))
    character = json.loads(response['messages'][-1].content)
    assert 'personality' in character
    assert type(character) is dict

def test_character_node_memory(llm):
    config = {'configurable': {'thread_id': '1'}}
    memory = MemorySaver()
    create_character = to_compiled_graph(create_character_node(llm), memory)
    message = "Create a character in a fantasy world. Make sure it's different from ones you've already created."
    response1 = create_character.invoke(human_message(message), config=config)['messages'][-1]
    response2 = create_character.invoke(human_message(message), config=config)['messages'][-1]
    condition = 'Respond SAME if the responses describe the same character in a story. Respond NOT SAME if they do not.'
    result = _are_similar_responses(llm, response1, response2, condition)
    assert result == 'NOT SAME', f'Got \n{response1.content[:35]}\n{response2.content[:35]}'


def test_create_character_node__as_graph(llm):
    create_character = create_character_node(llm)
    as_model = to_compiled_graph(create_character)
    response2 = as_model.invoke(human_message('A person who likes dogs.'))
    character = response2['messages'][-1]
    assert 'personality' in character.content, f'{response2}'

def test_create_setting_node(llm):
    create_setting = create_setting_node(llm)
    response: dict = create_setting(human_message('A place full of happy dogs.'))
    setting = json.loads(response['messages'][-1])
    assert 'physical_description' in setting
    assert type(setting) is dict

def test_create_setting_node__as_graph(llm):
    create_character = create_setting_node(llm)
    as_model = to_compiled_graph(create_character)
    response = as_model.invoke(human_message('A place full of happy dogs.'))
    setting = response['messages'][-1]
    assert 'physical_description' in setting.content, f'{response}'

def test_setting_node_memory(llm):
    config = {'configurable': {'thread_id': '1'}}
    memory = MemorySaver()
    create_setting = to_compiled_graph(create_setting_node(llm), memory)
    message = "Create a setting in a fantasy world. Make sure it's different from ones you've already created."
    response1 = create_setting.invoke(human_message(message), config=config)['messages'][-1]
    response2 = create_setting.invoke(human_message(message), config=config)['messages'][-1]
    condition = 'Respond SAME if the responses describe the same setting in a story. Respond NOT SAME if they do not.'
    result = _are_similar_responses(llm, response1, response2, condition)
    assert result == 'NOT SAME', f'Got \n{response1.content[:35]}\n{response2.content[:35]}'

def test_create_detail_node(llm):
    create_detail = create_detail_node(llm)
    response: dict = create_detail(human_message('Your topic is "Ways dogs make humans happy".'))
    details = json.loads(response['messages'][-1])
    assert 'details' in details
    assert type(details) is dict

def test_create_detail_node__as_graph(llm):
    create_detail = create_detail_node(llm)
    as_model = to_compiled_graph(create_detail)
    response = as_model.invoke(human_message('The detail is "Ways dogs make humans happy".'))
    details = response['messages'][-1]
    assert 'topic' in details.content, f'{response}'

def test_detail_node_memory(llm):
    config = {'configurable': {'thread_id': '1'}}
    memory = MemorySaver()
    create_detail = to_compiled_graph(create_detail_node(llm), memory)
    message = "Create a magic spell in a fantasy world. Make sure it's different from ones you've already created."
    response1 = create_detail.invoke(human_message(message), config=config)['messages']
    response2 = create_detail.invoke(human_message(message), config=config)['messages']
    assert len(response1) < len(response2), "With memory, the second response should include additional messages."
    condition = 'Respond SAME if the responses describe the same spell in a story. Respond NOT SAME if they do not.'
    result = _are_similar_responses(llm, response1[-1].content, response2[-1].content, condition)
    assert result == 'NOT SAME', f'Got \n{response1.content[:50]}\n{response2.content[:50]}'


@pytest.mark.parametrize('prompt, test_key', [
    ('Create a person who likes ice cream.', 'personality'),
    ('Create an ice cream shop.', 'physical_description'),
    ('Create a type of ice cream.', 'topic')
])
def test_writer_graph(prompt, test_key, llm):
    create_world = writer_graph(llm).compile()
    message = human_message(prompt)
    response = create_world.invoke(message)
    json_content = json.loads(response['messages'][-1].content)
    assert isinstance(json_content, dict)
    assert test_key in json_content
