import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import streamlit as st

load_dotenv()

# #vector db details
# ATLAS_CONNECTION_STRING = os.getenv("ATLAS_CONNECTION_STRING")
# DB_NAME = os.getenv("DB_NAME")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

ATLAS_CONNECTION_STRING = st.secrets["atlas_commection_string"]
DB_NAME = st.secrets["db_name"]
COLLECTION_NAME = st.secrets["collection_name"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = st.secrets["atlas_vector_search_index_name"]
GOOGLE_API_KEY = st.secrets["google_api_key"]
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["google_application_credentials"]

#Instantiate vector store
embeddings = VertexAIEmbeddings('text-embedding-005')
namespace = (DB_NAME + "." + COLLECTION_NAME)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
  connection_string = ATLAS_CONNECTION_STRING,
  namespace = namespace,
  embedding = embeddings,
  index_name = ATLAS_VECTOR_SEARCH_INDEX_NAME
)

#Define model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

#Initialize chat history
chat_history = []

#Configure prompt for system to use to reformulate original prompt after processing chat history
chathist_system_prompt = """Given a chat history and the latest user question which  might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
chathist_prompt = ChatPromptTemplate.from_messages(
    [("system",chathist_system_prompt),
     MessagesPlaceholder("chat_history"),
     ("human","{input}")
     ]
     )

#Configure main question and answer prompt
qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
{context} 
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [("system",qa_system_prompt),
     MessagesPlaceholder("chat_history"),
     ("human","{input}")
     ]
     )

#Define vector store as retriever
retriever = vector_store.as_retriever(search_type='similarity')

#Define chat history chain
history_aware_retriever = create_history_aware_retriever(model, retriever, chathist_prompt)

#Define main q&a chain
qa_chain = create_stuff_documents_chain(model, qa_prompt)


#Define the main RAG chain
def rag_response(query):
    rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)
    return rag_chain.invoke({"input":query,
                             "chat_history":chat_history})

#Define function that will execute RAG chain and record chat history
def query(query):
    response = rag_response(query)
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    print("history", response["chat_history"])
    return response




