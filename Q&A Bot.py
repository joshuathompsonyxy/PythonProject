import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


os.environ["OPENAI_API_KEY"] = "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" # Make sure this line has your actual key


DOCUMENT_PATH = "/mydoc.txt"


def setup_qa_chain():
    """
    Sets up the LangChain QA retrieval chain.
    """
    try:
   
        print(f"Loading document from: {DOCUMENT_PATH}")
        loader = TextLoader(DOCUMENT_PATH)
        documents = loader.load()
       
        print(f"Loaded {len(documents)} document(s).")

     
       
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print(f"Split document into {len(texts)} chunks.")

       
        print("Creating OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()

       
        print("Creating FAISS vector store...")
        docsearch = FAISS.from_documents(texts, embeddings)
        print("Vector store created.")

        
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

       
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
                
                response = qa_chain.invoke({"query": query})
                print(f"Bot: {response['result']}")
            except Exception as e:
                print(f"An error occurred while getting the answer: {e}")
                print("Please try again or check your API key/internet connection.")

    else:
        print("Failed to set up the QA chain. Exiting.")

if __name__ == "__main__":
    main()

       


