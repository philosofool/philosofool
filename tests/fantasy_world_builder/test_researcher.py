from fantasy_world_builder.researcher import create_research_tool, create_researcher, researcher_graph
from langchain_core.messages import HumanMessage

def test_create_research_tool(vector_db):
    research_tool = create_research_tool(vector_db)
    result = research_tool.invoke({'query': 'Who is a bartender?', 'n_documents': 1})
    assert 'Sam' in result[0], f"Got {result[0]}"
    assert len(result) == 1

    result2 = research_tool.invoke({'query': 'Who is a bartender?', 'n_documents': 2})
    assert len(result2) == 2

def test_create_research_tool_from_path():
    path = 'test_simple_db.vdb'
    research_tool = create_research_tool(path)
    assert hasattr(research_tool, 'invoke'), 'When getting the tool from path, the resulting tool should have invoke.'

def test_create_researcher(llm, vector_db):
    tool = create_research_tool(vector_db)
    llm_with_tools = llm.bind_tools([tool])
    researcher = create_researcher(llm_with_tools)
    response = researcher({'messages': [HumanMessage("Who likes trivia?")], 'routing': 'research'})
    tool_calls = response['messages'][-1].tool_calls
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call['name'] == 'research_tool'

def test_research_graph(llm, vector_db):
    agent = researcher_graph(llm, vector_db).compile()
    response = agent.invoke({'messages': [HumanMessage("Who likes trivia?")]})
    assert any(('Norm' in message.content for message in response['messages'])), f'Expected a document about Norm. Got {[message.content for message in response['messages']]}'
