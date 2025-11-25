from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I despise mangoes.',
    'I love linux.',
    'I hate windows.'
]

vector_store = FAISS.from_texts(texts, embedding = embeddings)
# print(vector_store.similarity_search('Apples are my favourite food.', k=7))
# print(vector_store.similarity_search('Linux is a great operating system.', k=7))

# results = vector_store.similarity_search('What the person dislikes?', k=4)


retriever = vector_store.as_retriever(search_kwargs={'k': 3})

@tool
def kb_search(query: str) -> str:
    """Search the small product / fruit knowledge base for information"""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

agent = create_agent(
    model = 'gpt-4o-mini',
    tools = [kb_search],
    system_prompt = (
        "You are a helpful assistan. For questions about Macs, apples, or laptops, "
                    "first call the kb_search tool to retrieve context, then answer succintly. Maybe you have to use it multiple times before answering"
    ),
    
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})

print(result)