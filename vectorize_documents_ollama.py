from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()


embeddings = OllamaEmbeddings(model="llama2")
def load_chunk_persis_txt():
    txt_folder_path = "./docs/Mtech_curriculum"
    documents = []
    for file in os.listdir(txt_folder_path):
        if file.endswith('.txt'):
            txt_path = os.path.join(txt_folder_path, file)
            loader = TextLoader(txt_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents = text_chunks,
        embedding=embeddings,
        persist_directory="vector_db_dir_ollama"
    )
    print("Documents vectorization completed !!")

def main():
    load_chunk_persis_txt()

if __name__ == "__main__":
    main()


    
# loading the embedding model

# loader = DirectoryLoader(path="./docs", glob="./*.pdf",
# loader_cls=UnstructuredFileLoader)

# documents = loader.load()




# -------------------------------------------------
# vectorize_document.py Contents


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# def load_chunk_persis_txt():
#     txt_folder_path = "./docs"
#     documents = []
#     for file in os.listdir(txt_folder_path):
#         if file.endswith('.txt'):
#             txt_path = os.path.join(txt_folder_path, file)
#             loader = TextLoader(txt_path)
#             documents.extend(loader.load())
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     text_chunks = text_splitter.split_documents(documents)
#     vectordb = Chroma.from_documents(
#         documents = text_chunks,
#         embedding=embeddings,
#         persist_directory="vector_db_dir_1000"
#     )
#     print("Documents vectorization completed !!")

def load_chunk_persis_txt():
    txt_folder_path = "./docs"
    documents = []
    for file in os.listdir(txt_folder_path):
        if file.endswith('.txt'):
            txt_path = os.path.join(txt_folder_path, file)
            print(f"Processing file: {txt_path}")
            loader = TextLoader(txt_path)
            docs = loader.load()
            for doc in docs:
                # add metadata to each document indicating its source
                doc.metadata = {"source": file}
                documents.append(doc)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    # display info about split document
    print("\n ---- document chunks information ----")
    print(f"Number of document chunks: {len(text_chunks)}")
    print(f"Chunks: {text_chunks[0].page_content}")
    vectordb = Chroma.from_documents(
        documents = text_chunks,
        embedding=embeddings,
        persist_directory="vector_db_dir_1000"
    )
    query = "List down syllabus for probability and stats."
    docs = vectordb.similarity_search(query)
    print(docs[0].page_content)
    print("Documents vectorization completed !!")


def main():
    load_chunk_persis_txt()

if __name__ == "__main__":
    main()


    
# loading the embedding model

# loader = DirectoryLoader(path="./docs", glob="./*.pdf",
# loader_cls=UnstructuredFileLoader)

# documents = loader.load()




