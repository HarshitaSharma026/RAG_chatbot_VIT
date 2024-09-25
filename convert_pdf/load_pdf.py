from langchain_community.document_loaders import PyPDFLoader
import asyncio
import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# converting pdf into text using pypdf loader
async def load_pdf():
    file_path = "./docs/Mtech_curriculum/information_security.pdf"
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    # print(f"{pages[2].metadata}")
    # print(pages[2].page_content)

    store = InMemoryVectorStore.from_documents(pages, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    docs = store.similarity_search("What are discipline core courses?", k = 2)
    for doc in docs:
        print(f'Page: {doc.metadata["page"]}: {doc.page_content[:20]}\n')

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_pdf())
    loop.close()
if __name__ == '__main__':
    main()