from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o"
)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="planilha_financeira"
)

retriever = vectorstore.as_retriever()

system_prompt = """
Use o contexto para responder às perguntas sobre a planilha financeira.

Contexto:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

query = "Faz uma lista de gastos do maior para o menor?"

response = chain.invoke(query)

print(response)
