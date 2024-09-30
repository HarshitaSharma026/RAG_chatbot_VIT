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




