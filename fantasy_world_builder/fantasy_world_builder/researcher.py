from __future__ import annotations

import textwrap
from typing import Annotated, TYPE_CHECKING
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from fantasy_world_builder.database import SimpleVectorDB
from fantasy_world_builder.writer import WriterState


def create_research_tool(database: SimpleVectorDB | str):
    if type(database) is str:
        database = SimpleVectorDB.from_path(database)

    @tool
    def research_tool(query: Annotated[str, "A query saying what to look for."], n_documents: Annotated[int, "The number of documents to retrieve."]):
        """Get n documents related to a query."""
        return database.get_documents(query, n_documents)

    return research_tool

def create_researcher(llm_with_tools):
    # tool = create_research_tool(database=database)
    # research_llm = llm.bind_tools([tool])

    def _parse_messages(messages: list):
        prompt = textwrap.dedent(
            """
            You are a researcher who collects information and returns it to an agent that uses the information.
            INSTRUCTIONS:
            - Do research on input data. Assume that research will tell you something you didn't expect.
            - Don't answer before you have done research.
            - Collect a number of records based on the number of entities or concepts mentioned in the user input.
            - Return the data that you find in its structured form. DON'T add any ideas of your own.
            """
        )
        last_message = messages[-1]
        if last_message.name == 'research_tool':
            return messages
        else:
            return [SystemMessage(prompt), last_message]

    def researcher(state: WriterState):
        state_input = _parse_messages(state['messages'])
        response = llm_with_tools.invoke(state_input)
        return {'messages': [response]}

    return researcher

def tool_routing(state: WriterState):
    last_messsage = state['messages'][-1]
    if last_messsage.name == 'research_tool':
        return END
    return 'tools'

def researcher_graph(llm, database: SimpleVectorDB | str) -> StateGraph:
    graph = StateGraph(WriterState)
    research_tool = create_research_tool(database)

    researcher = create_researcher(llm.bind_tools([research_tool]))
    tool_node = ToolNode([research_tool])

    graph.add_node('research', researcher)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'research')
    graph.add_edge('tools', END)
    graph.add_conditional_edges(
        'research',
        tool_routing
    )
    return graph