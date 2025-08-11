import textwrap
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from fantasy_world_builder.writer import WriterState

def routing(state: WriterState):
    last_task = state.get('routing')
    if last_task == 'research':
        return {'routing': 'write'}
    elif last_task == 'write':
        return {'routing': 'build'}
    elif last_task == 'build':
        return {'routing': 'finished'}
    else:
        return {'routing': 'research'}

# def create_supervisor(llm: BaseChatModel, prompt: str | None):
#     prompt = prompt or textwrap.dedent(
#         """
#         You are a supervisor who delegates to agents who perform tasks.
#         The project who coordinate creates content for fantasy worlds,
#         including setting, character and details.

#         Your role is to pass along messages that assist each step in the process.
#         """)

def supervisor(state: WriterState):
    task = state['routing']
    if task == 'research':
        last_message = state['messages'][-1]
        return {'messages': [SystemMessage(
            f"""Your most recent prompt is \n    {last_message.content}\nDo research on this prompt and send back the results of the research."""
        )]}
    if task == 'write':
        return {'messages': [SystemMessage("Use the available research and write on the most recent prompt.")]}
    if task == 'build':
        return {'messages': []}  # in expected configuration, the last message from the writer is all the builder needs. Superisor just needs to allow the messages to flow.
    if task == 'finished':
        return {'messages': [SystemMessage("An entity has been added to the world.")]}
    raise ValueError(f"Expected 'finished', 'research', 'write' or build' as the state routing. Got {task}")
