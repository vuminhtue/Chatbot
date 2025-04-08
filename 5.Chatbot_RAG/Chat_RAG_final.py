from langchain_community.chat_models import ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma 
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
import streamlit as st
import os
import shutil

# Define functions
def generate_response(chat_history):
    response = qa_chain.run(chat_history)
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

def split_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 150,
            add_start_index=True
        )
    return text_splitter.split_documents(doc)

# Design SideBar including title, information on llm model, temperature and images
st.sidebar.title("Settings")
llm_name = st.sidebar.selectbox("Model", ["llama3.2:1b","gemma3:1b","deepseek-r1:1.5b"])
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.5,0.01)
st.sidebar.image("./pony.jpeg")
st.title("Ask Perunas")


# Design the chatbot
llm = ChatOllama(base_url="http://localhost:11434", model=llm_name, temperature=temperature)
persist_directory = "./docs/chroma"
oembeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")
system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant. You explain things in short and brief. If you don't know or unsure about question, say I dont know")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=oembeddings)

update = st.sidebar.button("Update database!")
if update:
    with st.spinner("Updating database..."):
    # Load PDF
        pdf_files = [f for f in os.listdir("./pdf/") if f.endswith(".pdf")]
        pdf_loader = [PyPDFLoader("./pdf/"+i) for i in pdf_files]

        docs = []
        for loader in pdf_loader:
            docs.extend(loader.load())

        splits = split_text(docs)

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=oembeddings,
            persist_directory=persist_directory
        )
        st.write("Database updated successfully!!!")


    # Design Q&A
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
qtext = st.chat_input("What do you want to ask from PDF?")

if qtext:
    st.write(f"**:adult: user**: {qtext}")
    with st.spinner("Thinking..."):
        prompt = HumanMessagePromptTemplate.from_template(qtext)
        chat_history = get_history()
        chat_history.append(prompt)
        response = generate_response(qtext)
        st.write(f"**:horse: peruna**: {response}")
        st.write("---")
        st.session_state['chat_history'].append({'user': qtext, 'assistant': response})

for chat in reversed(st.session_state['chat_history']):
        st.write(f"**:adult: User**: {chat['user']}")
        st.write(f"**:horse: Peruna**: {chat['assistant']}")
            