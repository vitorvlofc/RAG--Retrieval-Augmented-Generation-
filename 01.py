from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Carrega as variáveis de ambiente
load_dotenv()

PDF_PATH = "Manual_notebook.pdf"
CHROMA_PATH = "./chroma_db"


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


# Cria os embeddings
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


# Cria o banco vetorial
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="manual_notebook",
    persist_directory=CHROMA_PATH
)


# Cria o retriever
retriever = vectorstore.as_retriever()


# Realiza a busca
result = retriever.invoke(
    "quanto de memória RAM o notebook possui?"
)


print(f"Resultado da busca: {result}")
