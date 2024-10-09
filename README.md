# :robot: Multidocument RAG-based Conversational Chatbot
This chatbot project is specifically designed for educational institutions (my own university for now) to answer students queries based on a large set of documents using a Retrieval-Augmented Generation (RAG) approach. The chatbot retrieves relevant documents, processes user inputs while maintaining session history, and formulates accurate responses using the context from the documents. It utilizes a history-aware retriever to maintain the chat flow and provide detailed responses based on prior questions or user references. 

Its an ongoing project done individually, that is why the use cases covered are limited. For now the chatbot can answer queries related to university's:
- Academic calendar 
- General FAQs
- Exam related FAQs
- Curriculum (syllabus) information - only for MCA and MTech courses provided by university.
- Facult information (only covering facult of MCA for now)

## :technologist: Technologies Used
1. LangChain, for implementing retrieval-based pipelines.
2. Chroma, for document vector storage and similarity search.
3. Streamlit, for creating the chatbot interface.
4. Google Generative AI API, for embeddings and question-answering.
5. Python, for all backend operations.
6. dotenv, for API key management.

## :technologist: Installation guide 
1. Clone the repo 
```
git clone https://github.com/<your-name>/RAG_chatbot_VIT.git
```
2. Navigate to project directory
```
cd RAG_chatbot_vit
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Create a .env file to set up your environment variable 
```
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```
5. Run the project: Start the Streamlit app by running the following command in your terminal:
```
streamlit run 03_main.py
```

## Project Structure
**convert_pdf** : contains files to convert the pdf document into raw text documents or md files. This conversion is essential because a pdf contains a lot of unstrctured and complex data, which is difficult to convert into vectors and difficult for llm model to understand. That is why we need to convert it into simple text, or md file. 

**docs** : the directory which contains all the text documents that are used as knowledge base of this chatbot.

**dummy** : same as docs, but this directory is being referred in **embeddings.py** to convert embedding of text files.

**dummydb** : hold the vector embeddings of all the text documents.

**03_main.py** : the main applications file to launch the chatbot using streamlit

**concepts.md** : file that contains explaination of all the new concepts that I can across while learning to develop the chatbot. More concepts will be added to this file as the project progresses.

**embeddings.py** : file that is converting text documents to their vector formats. Uses Chroma db to store the embedding.

**problems_faced.md** : consists of the list of problems I faced, and how I solved it, including link of resources used to solve the problem. 

**requirements.txt** : consists of all the dependencies required to build the chatbot

## How to Use the Chatbot
- Refer to the text documents in "dummy" directory to get an idea of what type of questions (from the documents) can be asked. 
- Enter your query in the input box.
- The chatbot will retrieve relevant documents and provide detailed answers based on the context.
- The chat history is maintained throughout the session to ensure continuity in conversation.


**:dart:This is an ongoing project. I am actively learning and building side by side making it give more accurate answers. So feel free to contribute, report issues, or suggest enhancements to help improve its performance.**