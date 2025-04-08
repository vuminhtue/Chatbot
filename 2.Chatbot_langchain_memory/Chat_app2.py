from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
import streamlit as st

# Design SideBar including title, information on llm model, temperature and images
st.sidebar.title("Settings")
llm_name = st.sidebar.selectbox("Model", ["llama3.2:1b","gemma3:1b","deepseek-r1:1.5b"])
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.5,0.01)
st.sidebar.image("./pony.jpeg")
st.title("AskPeruna")

llm = ChatOllama(base_url="http://localhost:11434", model=llm_name, temperature=temperature)
system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant. You explain things in short and brief. If you don't know or unsure about question, say I dont know")

# Function to generate response from inserted text:

def generate_response(chat_history):
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template|llm|StrOutputParser()
    response = chain.invoke({})
    return response

def get_history():
    chat_history = [system_message]
    for chat in st.session_state['chat_history']:
        prompt = HumanMessagePromptTemplate.from_template(chat['user'])
        chat_history.append(prompt)

        ai_message = AIMessagePromptTemplate.from_template(chat['assistant'])
        chat_history.append(ai_message)

    return chat_history

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# Design the main text box
qtext = st.chat_input("What do you want to ask?")

if qtext:
    with st.spinner("Generating response..."):
        prompt = HumanMessagePromptTemplate.from_template(qtext)
        chat_history = get_history()
        chat_history.append(prompt)
        response = generate_response(chat_history)
        st.session_state['chat_history'].append({'user': qtext, 'assistant': response})

for chat in reversed(st.session_state['chat_history']):
       st.write(f"**:adult: User**: {chat['user']}")
       st.write(f"**:horse: Peruna**: {chat['assistant']}")
       st.write("---")