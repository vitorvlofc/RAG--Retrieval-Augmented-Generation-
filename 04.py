from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

XLSX_PATH = "Planilha_Financeira_Vitor.xlsx"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "planilha_financeira"

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

loader = UnstructuredExcelLoader(
    file_path=XLSX_PATH,
    mode="elements"
)

documents = loader.load()

print(f"Total de documentos: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total de chunks: {len(chunks)}")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH
)

print("Dados inseridos no banco vetorial.")
print(f"Documentos no Chroma: {vectorstore._collection.count()}")
