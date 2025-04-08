from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate
import streamlit as st

# Design SideBar including title, information on llm model, temperature and images
st.sidebar.title("Settings")
llm_name = st.sidebar.selectbox("Model", ["llama3.2:1b","gemma3:1b","deepseek-r1:1.5b"])
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.5,0.01)
st.sidebar.image("./pony.jpeg")
st.title("AskPeruna")

llm = ChatOllama(base_url="http://localhost:11434", model=llm_name, temperature=temperature)


# Function to generate response from inserted text:
def generate_response(qtext):
    response = llm.invoke(qtext)
    return response.content

qtext = st.chat_input("What do you want to ask?")

if qtext:
    with st.spinner("Thinking..."):
        response = generate_response(qtext)
        st.write(response)
