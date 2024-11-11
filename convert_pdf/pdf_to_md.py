import nest_asyncio
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
load_dotenv()
nest_asyncio.apply()


dir_path = "/Users/harshitawork/Desktop/RAG_chatbot_VIT/docs/pdf_docs"
for file in os.listdir(dir_path):
    if file.endswith(".pdf"):
        complete_file_path = os.path.join(dir_path, file)
        docs = LlamaParse(result_type="markdown").load_data(complete_file_path)

        new_filename = file.replace(".pdf", ".md")
        with open(new_filename, 'a') as file:
            for doc in docs:
                file.write(doc.text) 
