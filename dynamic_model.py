from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

basic_model = init_chat_model(model='gpt-4o-mini')
advanced_model = init_chat_model(model ='gpt-4.1-mini')

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'])

    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model

    request.model = model

    return handler(request)

agent = create_agent(
    model = basic_model,
    middleware = [dynamic_model_selection],
)

response = agent.invoke({
    'messages':[
        SystemMessage('You are a helpful assistant.'),
        HumanMessage('What is 5+5?'),
        HumanMessage('What is Langchain'),
        HumanMessage('What is RAG'),
        HumanMessage('What is Apple')
    ]
})

print(response['messages'][-1].content)
print(response['messages'][-1].response_metadata['model_name'])