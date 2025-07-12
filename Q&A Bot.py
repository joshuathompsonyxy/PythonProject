import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# IMPORTANT: Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key.
# You can get one from https://platform.openai.com/account/api-keys
# It's recommended to set this as an environment variable for security.
# For demonstration, we'll set it directly here, but for production, use os.environ.
os.environ["OPENAI_API_KEY"] = "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" # Make sure this line has your actual key

# Define the path to your document
# Based on your screenshot, it looks like you're using an absolute path for mydoc.txt
# If mydoc.txt is directly in your PythonProject folder, you can use:
# DOCUMENT_PATH = "./mydoc.txt"
# If it's inside the .venv folder as your screenshot implies, keep your current path:
DOCUMENT_PATH = "/mydoc.txt"


def setup_qa_chain():
    """
    Sets up the LangChain QA retrieval chain.
    """
    try:
        # 1. Load the document
        print(f"Loading document from: {DOCUMENT_PATH}")
        loader = TextLoader(DOCUMENT_PATH)
        documents = loader.load()
        # FIX: Ensure the closing parenthesis is correctly placed
        print(f"Loaded {len(documents)} document(s).")

        # 2. Split the document into chunks
        # This helps the LLM process long documents by breaking them into smaller, manageable pieces.
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print(f"Split document into {len(texts)} chunks.")

        # 3. Create embeddings
        # Embeddings convert text into numerical vectors, allowing for semantic search.
        print("Creating OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()

        # 4. Create a vector store (FAISS in this case)
        # The vector store stores the document chunks and their embeddings, enabling efficient retrieval.
        print("Creating FAISS vector store...")
        docsearch = FAISS.from_documents(texts, embeddings)
        print("Vector store created.")

        # 5. Initialize the Language Model (LLM)
        # We'll use ChatOpenAI for conversational capabilities.
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

        # 6. Create the RetrievalQA chain
        # This chain combines the LLM with the vector store to answer questions.
        # 'stuff' chain type puts all retrieved documents into the prompt.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever()
        )
        print("QA chain setup complete.")
        return qa_chain

    except Exception as e:
        print(f"An error occurred during setup: {e}")
        print("Please ensure your OpenAI API key is correct and you have an active internet connection.")
        return None

def main():
    """
    Main function to run the Q&A bot.
    """
    qa_chain = setup_qa_chain()

    if qa_chain:
        print("\n--- Document Q&A Bot ---")
        print("Type 'exit' or 'quit' to end the conversation.")

        while True:
            query = input("\nYour question: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting bot. Goodbye!")
                break

            try:
                # Get the answer from the QA chain
                response = qa_chain.invoke({"query": query})
                print(f"Bot: {response['result']}")
            except Exception as e:
                print(f"An error occurred while getting the answer: {e}")
                print("Please try again or check your API key/internet connection.")

    else:
        print("Failed to set up the QA chain. Exiting.")

if __name__ == "__main__":
    main()

       


