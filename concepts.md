

## Create embedding of documents
### raw_doc = TextLoader("docs/mca_syllabus.txt").load()
TextLoader() : class used to load the contents of a text file
load(): method will read the file and return a document object or list of text-based documents 

### text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
CharacterTextSplitter(), RecursiveCharacterTextSplitter(): classes are used to split the documents. It will return the respective text splitter object, that will be used to make chunks of all the documents, as specified.

### text_chunks = text_splitter.split_documents(raw_doc)
takes the raw_doc and splits it into chunks of 1000, with 500 overlap.
text_chunks = variable now contains "list" of text chunks

### db = Chroma.from_documents(documents, embeddings)
db = the created Chroma vector store which will hold embeddings for all doc chunks
Chroma.from_documents(): for each text chunk (stored in documents) create embeddings using the embedding mentioned.

### docs = db.similarity_search(query)
docs = list of most relevent documents (from chroma db) 
docs[0].page_content = will display the page content of documents / vector embedding most similar to be the query

### for file in os.listdir(document_path)
os.listdir():  is a Python method used to obtain a list of files and directories in a specified path or the current working directory.

### documents.append(raw_doc_list) & documents.extend(raw_doc_list)


## Creating retrievers 
### genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
Configures Google Generative AI with the API key.

### retriever = vectorstore.as_retriever() V/S matched_docs = vectordb.similarity_search(query)
similarity_search(): a similarity search directly in the vector database (like Chroma). It compares the query's embedding vector with the vectors of the documents stored in the database. The closer the vectors are in the vector space, the more similar the query and document are considered to be. It doesn't take into account any chat history or previous interactions. The query is treated in isolation, which can lead to less accurate retrieval if the query depends on prior context (e.g., follow-up questions).

retrievers: It can use the vector database under the hood but adds additional layers of processing. For example, retrievers can take into account chat history (context), rephrase questions, and use more advanced techniques to better understand the user's intent. context-aware, advanced ranking 

### MessagePlaceholder("history")
A placeholder which can be used to pass in a list of messages. (here passing history of the chats)

### create_history_aware_retriever 
is a function that creates a retriever capable of considering both current user input and the chat history, and getting relevant documents

### create_stuff_documents_chain
Create a chain for passing a list of Documents to a model, extracts information for documents into meaningful response.

### create_retrieval_chain
This function connects the retriever and the question-answering chain. It creates a RAG chain that first retrieves documents and then uses them to generate a detailed, accurate answer.

## Using streamlit

### Session state
provides a session state to store values that persist across reruns of the app. This is important for a chatbot because you need to retain the conversation history, vector store, and chat chain across different user interactions.

### Conversational Chain
The conversational chain is the pipeline responsible for:
- Retrieving relevant documents using the vector store (through the retriever).
- Passing the documents to a language model, which then uses them to answer the user's query.

### st.chat_message()
function is used to create a chat message in the Streamlit app, and it accepts a role argument.

### st.markdown() 
renders the text in the app using Markdown, which is a lightweight markup language for formatting text.