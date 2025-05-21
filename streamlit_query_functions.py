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
import tempfile

load_dotenv()

# #vector db details
# ATLAS_CONNECTION_STRING = os.getenv("ATLAS_CONNECTION_STRING")
# DB_NAME = os.getenv("DB_NAME")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

#Set API keys from st.secrets except for VertexAI keys
ATLAS_CONNECTION_STRING = st.secrets["atlas_connection_string"]
DB_NAME = st.secrets["db_name"]
COLLECTION_NAME = st.secrets["collection_name"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = st.secrets["atlas_vector_search_index_name"]
GOOGLE_API_KEY = st.secrets["google_api_key"]

##Handle Vertex AI secrets
def setup_gcp_credentials():
    # Check if running locally (with .env) or on Streamlit Cloud (with st.secrets)
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.info("Using GCP credentials from GOOGLE_APPLICATION_CREDENTIALS env variable.")
        # If running locally and GOOGLE_APPLICATION_CREDENTIALS is set, proceed
        # No need to do anything further here as the environment variable is already set

    elif hasattr(st, 'secrets') and 'google_application_credentials' in st.secrets:
        st.info("Using GCP credentials from Streamlit secrets.")
        try:
            # Get the JSON string from secrets
            service_account_info = st.secrets["google_application_credentials"]

            # Write the JSON string to a temporary file
            # This is crucial because google-auth expects a file path
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
                temp_file.write(service_account_info)
                temp_file_path = temp_file.name

            # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

            st.success(f"GCP credentials set from Streamlit secrets. Temp file: {temp_file_path}")

        except Exception as e:
            st.error(f"Error setting up GCP credentials from secrets: {e}")
            st.stop()
    else:
        st.warning("GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or 'gcp_service_account_json' in secrets) not found. Vertex AI Embeddings may fail.")
        # If running locally without a .env file, this will stop here
        # st.stop() # Uncomment if you want to strictly enforce credentials

setup_gcp_credentials()

# --- Initialize VertexAIEmbeddings ---
try:
    # VertexAIEmbeddings will now automatically pick up the GOOGLE_APPLICATION_CREDENTIALS
    # from the environment variable we just set.
    embeddings = VertexAIEmbeddings('text-embedding-005') # Specify your desired model
    st.success("VertexAIEmbeddings initialized successfully!")

except Exception as e:
    st.error(f"Failed to initialize VertexAIEmbeddings. Check your GCP credentials and permissions: {e}")
    st.stop()

#Instantiate vector store
#embeddings = VertexAIEmbeddings('text-embedding-005')
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




