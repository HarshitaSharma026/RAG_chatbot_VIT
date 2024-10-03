from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# LOAD A SINGLE FILE, AND QUERY ON IT 
# raw_doc = TextLoader("docs/ai_ml.txt").load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
# documents = text_splitter.split_documents(raw_doc)
# print(len(documents))
# # for ind, doc in enumerate(documents, 1):
# #     print(f"--------- Chunks : {ind} \n Doc data: {doc.page_content}")
# db = Chroma.from_documents(documents, embeddings)

# query = "List down syllabus for Français Fonctionnel"
# docs = db.similarity_search(query)
# for ind, doc in enumerate(docs):
#     print(f"Document {ind}\n Contents: \n {doc.page_content}")

# LOADING MULTIPLE FILES-----------------------------
documents = []         # hold the text of all the docs present in the directory
dir_path = "./docs"
for file in os.listdir(dir_path):
    if file.endswith(".txt"):
        complete_file_path = os.path.join(dir_path, file)
        raw_doc_list = TextLoader(complete_file_path).load()
        # append the raw document list to the external documents[] list
        # documents = [[mca1_doc], [ai_ml_doc], [cse_doc]]
        documents.extend(raw_doc_list)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)
vectordb = Chroma.from_documents(
    documents=text_chunks, 
    embedding=embeddings,
    persist_directory="./dummydb"
)

query = "What is the syllabus of probability and stats for MCA curriculum?"
matched_docs = vectordb.similarity_search(query)
for ind, doc in enumerate(matched_docs):
    print(f"-------- Document {ind}\nContents:\n{doc.page_content}")



