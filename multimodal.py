from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model('gpt-4o-mini')

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image', 'url': 'https://www.sciencelearn.org.nz/_next/image?url=https%3A%2F%2Fwww.datocms-assets.com%2F117510%2F1722402042-art_artificial_intelligence_neural_network_explain-281-29.png%3Fw%3D1840%26h%3D1270.9422492401216&w=1920&q=85'},
    ]
}

response = model.invoke([message])

print(response.content)

