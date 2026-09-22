from pathlib import Path

from dotenv import load_dotenv

from langsmith import Client
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Carrega as variáveis de ambiente
load_dotenv()

PDF_PATH = "Manual_notebook.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "manual_notebook"


# Cria os embeddings
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


# Verifica se o banco vetorial já existe
if Path(CHROMA_PATH).exists():

    print("Carregando banco vetorial existente...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=CHROMA_PATH
    )

else:

    print("Banco vetorial não encontrado. Criando...")

    # Carrega o PDF
    loader = PyMuPDF4LLMLoader(
        PDF_PATH,
        mode="page"
    )

    documents = loader.load()

    print(f"Total de páginas: {len(documents)}")


    # Divide o documento em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total de chunks: {len(chunks)}")


    # Cria o banco vetorial
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print("Banco vetorial criado com sucesso.")


# Cria o retriever
retriever = vectorstore.as_retriever()


# Carrega o prompt do LangChain Hub
client = Client()

prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True
)


# Cria o modelo
model = ChatOpenAI(
    model="gpt-4o-mini"
)


# Cria a cadeia RAG
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


# Faz as perguntas
try:

    while True:

        question = input("Digite sua pergunta ou aperte Ctrl+C para sair: ")

        result = rag_chain.invoke(question)

        print(f"Resposta: {result}")

except KeyboardInterrupt:

    print("\nEncerrando o programa.")
