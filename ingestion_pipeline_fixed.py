import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")


def load_documents(directory_path):
    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not list(directory_path.glob("*.txt")):
        raise FileNotFoundError(f"No .txt files found in: {directory_path}")

    loader = DirectoryLoader(
        str(directory_path),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts


def create_vector_store(texts):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or set it in your shell."
        )

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def main():
    documents = load_documents(DOCS_DIR)
    texts = split_documents(documents)
    vector_store = create_vector_store(texts)
    print(f"Vector store created successfully with {len(texts)} chunks!")


if __name__ == "__main__":
    main()
