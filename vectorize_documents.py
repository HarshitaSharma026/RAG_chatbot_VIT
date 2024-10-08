from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

documents = []         # hold the text of all the docs present in the directory
dir_path = "./docs"
for file in os.listdir(dir_path):
    if file.endswith(".txt"):
        complete_file_path = os.path.join(dir_path, file)
        raw_doc_list = TextLoader(complete_file_path).load()
        print(f"----------- File name: {complete_file_path}, chunks made: {len(raw_doc_list)}")
        # append the raw document list to the external documents[] list
        # documents = [[mca1_doc], [ai_ml_doc], [cse_doc]]
        documents.extend(raw_doc_list)
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)
print("Total chunks: ", len(text_chunks))
vectordb = Chroma.from_documents(
    documents=text_chunks, 
    embedding=embeddings,
    persist_directory="./vectordb"
)
print(f"Document vectorization complete !!")

query = "Give me syllabus for Data Structues in mtech cse?"
matched_docs = vectordb.similarity_search(query)
for ind, doc in enumerate(matched_docs):
    print(f"-------- Document {ind}\nContents:\n{doc.page_content}")