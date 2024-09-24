from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import asyncio
import os
load_dotenv()

# converting pdf into text using pypdf loader
# async def load_pdf():
#     file_path = "./docs/Mtech_curriculum/information_security.pdf"
#     loader = PyPDFLoader(file_path)
#     pages = []
#     async for page in loader.alazy_load():
#         pages.append(page)
#     # print(f"{pages[2].metadata}")
#     # print(pages[2].page_content)

#     store = InMemoryVectorStore.from_documents(pages, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
#     docs = store.similarity_search("What are discipline core courses?", k = 2)
#     for doc in docs:
#         print(f'Page: {doc.metadata["page"]}: {doc.page_content[:20]}\n')

# def main():
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(load_pdf())
#     loop.close()
# if __name__ == '__main__':
#     main()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def load_chunk_persis_pdf():
    pdf_folder_path = "./docs/Mtech_curriculum"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents = text_chunks,
        embedding=embeddings,
        persist_directory="vector_db_dir"
    )
    print("Documents vectorization completed !!")

def main():
    load_chunk_persis_pdf()

if __name__ == "__main__":
    main()


    
# loading the embedding model

# loader = DirectoryLoader(path="./docs", glob="./*.pdf",
# loader_cls=UnstructuredFileLoader)

# documents = loader.load()




