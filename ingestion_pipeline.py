import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_MAX_TOKENS = 512
CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50


def load_documents(docs_path):
    """Load all text files from the docs directory"""
   
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
   
    for i, doc in enumerate(documents[:3]):  # Show first 2 documents
        preview = doc.page_content[:100].replace("\ufeff", "")
        preview = preview.encode("cp1252", errors="replace").decode("cp1252")
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {preview}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def safe_console_text(text):
    """Return text that can be printed in the default Windows console."""
    return text.replace("\ufeff", "").encode("cp1252", errors="replace").decode("cp1252")


def split_documents(documents, chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS):
    """Split documents into chunks small enough for the embedding model."""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Estimated tokens: {text_splitter._length_function(chunk.page_content)}")
            print(f"Content:")
            print(safe_console_text(chunk.page_content))
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = TogetherEmbeddings(model=EMBEDDING_MODEL)
    
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    documents = load_documents("docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
