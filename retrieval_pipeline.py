from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = TogetherEmbeddings(model="intfloat/multilingual-e5-large-instruct")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")



# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatTogether model
model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

chat_history = []

def ask_question(user_question):
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Retrieve documents for the (possibly rewritten) search question
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        answer = ask_question(question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    start_chat()

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
