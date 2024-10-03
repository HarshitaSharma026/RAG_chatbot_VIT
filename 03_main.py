import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
# setting up working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

def setup_vector_store():
    persist_directory = f"{working_dir}/vector_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory,
    embedding_function = embeddings)

    return vectorstore

# create chain
def chat_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vectorstore.as_retriever()

# this prompt is to instruct the llm to reconstruct the question as a standalone question if the user has asked a question referening to previous question, or answer. "what is the syllabus of it?", so here "it" will be replaced with the actual course
    # This prompt is to reformulate the question into a standalone one
    rephrase_question_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    # here passing rephrase prompt + history of the chat + current user input
    repharse_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rephrase_question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Creating RETRIEVER for storing history, and retrieving documents
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, repharse_prompt
    )

    # System prompt for actual question answering
    system_prompt = (
        "Answer the question as detailed as possible from the documents that's provided to you, make sure to provide all the details. "
        "Cross-question the user if needed to clarify the question further so that you can answer it properly. "
        "If the user asks you about listing syllabus for any subject, just provide what's present is the knowledge base provided to you. Do not explain each of the topics in the syllabus unless the asks you to. Just provide the whole syllabus as it is mentioned in the knowledge base."
        "The link to the original document is provided in the document itself. "
        "Provide the link to the user if you're unable to get the complete answer. "
        "If the answer is not in the provided context just say, 'answer is not available in the context', "
        "DO NOT provide the wrong answer.\n\n"
        "Context:\n {context}?\n"
        "Question:\n {input}\n"
        "Answer:"
    )
    # Chat prompt template for the answer generation
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # creating QA chain to extract relevent documents
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # RAG chain = retriever + qa chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

st.set_page_config(
    page_title = "Multichat chatbot",
    page_icon="📑",
    layout="centered"
)

st.title("📑 Multi document chatbot")

# session state in streamlit
# when the user is using the app, that time all the history will be stored there, but as soon as the user presses refresh the history of the last session will be lost and new session will be created.

# Initialize session state for chat history if its not already there
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# chat_history = [{"user": "how are you?"}, {"assistant", "i am good, i am ai"}]

# check if session_state already has "vectorstore" variable, if not it'll call the setup_vector_store() method to intialize the vector store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vector_store()

# checks if it already has "cconversational_chain" variable, if not it'll call chat_chain(vectorstore) method and pass vector stores to create the chain.
# Initializing the conversational_chain here ensures that the chatbot can process user queries by retrieving relevant information and formulating coherent, informed responses.
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)


# Display the history of messages in the current session
# chat_history : stores messages as dictionaries, with each dictionary containing a role (either "user" or "assistant") and the corresponding content (the actual text of the message).
# ------ This code is responsible for displaying the conversation history in the Streamlit app.  -------
for message in st.session_state.chat_history:
    # This line sets up a context for displaying the message in the chat format, with the appropriate role.
    with st.chat_message(message["role"]):
    # This line displays the actual text content of the message using the st.markdown() function.
    # The message["content"] retrieves the actual text of the message (either the user's input or the chatbot's response).
    # st.markdown() renders the text in the app using Markdown, which is a lightweight markup language for formatting text.
    # It displays the message content in the chat interface with any Markdown-supported formatting (such as bold, italics, or lists).
        st.markdown(message["content"])

# Input for user to ask questions
user_input = st.chat_input("Ask AI...")

# Handle user input
if user_input:
    # Add the user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the response from the conversational chain
    with st.chat_message("assistant"):
        # calling conversational_chain, while also passing user_input and chat_history
        response = st.session_state.conversational_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})

        # The response from the model is retrieved using the "answer" key, and it is then rendered using st.markdown().
        assistant_response = response["answer"]
        st.markdown(assistant_response)

# Document(metadata={'source': './docs/academic_calender.txt'}, page_content="......"})
        print(response['source'])
       

        # Add the assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
