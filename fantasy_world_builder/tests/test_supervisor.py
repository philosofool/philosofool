import pytest
from langchain_core.messages import HumanMessage
from fantasy_world_builder.supervisor import supervisor, routing

@pytest.mark.parametrize('routing, expected', [
    ('research', 'Create a brewery in the fantasy town of Lilac Valley.'),
    ('write', "Use the available research and write on the most recent prompt."),
    ('build', ''),
    ('finished', "An entity has been added to the world.")
])
def test_supervisor(routing, expected):
    # supervisor = create_supervisor(llm)
    state = {'routing': routing, 'messages': [HumanMessage('Create a brewery in the fantasy town of Lilac Valley.')]}
    response = supervisor(state)
    if expected:
        content = response['messages'][-1].content
        assert expected in content
    else:
        assert response['messages'] == []

def test_supervisor_routing():
    state = {}
    state = routing(state)
    assert state['routing'] == 'research'
    state = routing(state)
    assert state['routing'] == 'write'
    state = routing(state)
    assert state['routing'] == 'build'
    state = routing(state)
    assert state['routing'] == 'finished'
    state = routing(state)
    assert state['routing'] == 'research'
