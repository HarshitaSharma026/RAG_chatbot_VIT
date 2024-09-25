import os
import json
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
# setting up working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

def setup_vector_store():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory,
    embedding_function = embeddings)

    return vectorstore

# create chain
def chat_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm = model,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    return chain

st.set_page_config(
    page_title = "Multichat chatbot",
    page_icon="📑",
    layout="centered"
)

st.title("📑 Multi document chatbot")

# session state in streamlit
# when the user is using the app, that time all the history will be stored there, but as soon as the user presses refresh the history of the last session will be lost and new session will be created.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vector_store()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)


# here we are displaying the history of all the messages in one session
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask ai...")

if user_input:
    # this line is just adding the question that is asked by the user as a dictionary in the chat_history = [] list that is created on line 53
    st.session_state.chat_history.append({"role":"user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # we get repsonse of the user question, answer will be stored in response variable
        response = st.session_state.conversational_chain({"question":user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)

        # adding chatbot response to chat history
        st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})

