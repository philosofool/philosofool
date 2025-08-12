
from __future__ import annotations

from collections.abc import Callable
import json
import textwrap
from typing import Literal

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from fantasy_world_builder.schema import Character, Setting, Detail, WriterState


def create_writer_node(llm: BaseChatModel) -> Callable[[WriterState], dict]:
    instructions = textwrap.dedent(
        """
        You delegate the role of writer to an agent specialized for writing:
        - CHARACTER: an entities with thoughts, personalities and agency.
        - SETTINGS: a place where events take places.
        - DETAILS: Not a place or a person; an event, story, etc.
        reply with CHARACTER, SETTING or DETAIL.
        Does this prompt:\n{}\nsuggest writing a CHARACTER, SETTING or DETAIL?"
        """
    )

    def writer_node(state: WriterState) -> dict[Literal['routing'], str]:
        prompt = state['messages'][-1].content
        message = instructions.format(textwrap.indent(prompt, '    '))
        response = llm.invoke([SystemMessage(message)])
        return {'routing': response.content}    # pyright: ignore [reportReturnType]

    return writer_node

def create_character_node(llm: BaseChatModel) -> Callable[[WriterState], dict]:
    llm = llm.with_structured_output(Character)    # pyright: ignore [reportAssignmentType]

    def character_node(state: WriterState) -> dict[Literal['messages'], list]:
        user_input = state['messages']
        prompt = SystemMessage("You are an agent whose job is to write descriptions of characters.")
        response = llm.invoke(user_input[:-1] + [prompt] + user_input[-1:])
        return {'messages': [AIMessage(json.dumps(response))]}
    return character_node

def create_setting_node(llm: BaseChatModel) -> Callable[[WriterState], dict]:
    def setting_node(state: WriterState) -> dict[Literal['messages'], list]:
        user_input = state['messages']
        prompt = SystemMessage("You are an agent whose job is to write descriptions of settings.")
        response = llm.with_structured_output(Setting).invoke(user_input[:-1] + [prompt] + user_input[-1:])
        return {'messages': [AIMessage(json.dumps(response))]}
    return setting_node

def create_detail_node(llm: BaseChatModel) -> Callable[[WriterState], dict]:
    def detail_node(state: WriterState) -> dict:
        user_input = state['messages']
        prompt = SystemMessage("You are an agent whose job is to write descriptions of subjects, events or other things that are not people or places.")
        response = llm.with_structured_output(Detail).invoke(user_input[:-1] + [prompt] + user_input[-1:])
        return {'messages': [AIMessage(json.dumps(response))]}
    return detail_node

def _routing(state: WriterState) -> str:
    route = state['routing']
    return route.lower() + '_node'

def writer_graph(llm: BaseChatModel) -> StateGraph:

    graph = StateGraph(MessagesState)
    graph.add_node('character_node', create_character_node(llm))    # pyright: ignore [reportArgumentType]
    graph.add_node('setting_node', create_setting_node(llm))  # pyright: ignore [reportArgumentType]
    graph.add_node('detail_node', create_detail_node(llm))  # pyright: ignore [reportArgumentType]
    graph.add_node('writer_node', create_writer_node(llm))  # pyright: ignore [reportArgumentType]

    graph.add_edge(START, 'writer_node')
    graph.add_conditional_edges(
        'writer_node',
        _routing
    )
    graph.add_edge('character_node', END)
    graph.add_edge('setting_node', END)
    graph.add_edge('detail_node', END)
    return graph