import os 
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

# define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "books", "os.txt")
persistent_directory = os.path.join(current_dir, "db","chroma_db_is")

# define embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# load the existing vector store with the embedding function
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings)

query = "List down all dicipline core subjects."

# retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3, "score_threshold":0.3},
)
relevant_docs = retriever.invoke(query)


# IMP POINT ********
# while building rag documents, if we are not getting results even after everything is working fine, it means we are getting too strict in retrieving the relevant documents 

# display documents relevant to the query
print("\n --- Relevant documents --- ")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")