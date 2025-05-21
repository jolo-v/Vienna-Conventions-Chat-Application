import streamlit as st
from dotenv import load_dotenv
from streamlit_query_functions import query

#load_dotenv()

#Page title
st.title("Vienna Conventions Chat App")
chat_placeholder = st.empty()

#Initilaize chat history with system message
def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state.messages = [{"role":"system","content":"You are a helpful assitant."}]

#Start the chat
def start_chat():
    #Display chat messages from history
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    #Accept user input
    if prompt := st.chat_input("This chatbot is here to answer your questions about the Vienna Conventions of 1961, 1963, and 1969."):
        #Add message to chat history
        st.session_state.messages.append({"role":"user","content":prompt})
        #Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt)
        #Generate response from LLM
        response = query(prompt)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
        #Add LLM's response to chat history
        st.session_state.messages.append({"role":"assistant","content":response["answer"]})

if __name__ == "__main__":
    init_chat_history()
    start_chat()


