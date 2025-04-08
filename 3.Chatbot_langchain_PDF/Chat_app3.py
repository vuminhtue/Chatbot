from langchain_community.chat_models import ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma 
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

import streamlit as st

# Define functions
def generate_response(qtext):
    response = qa_chain.run(qtext)
    return response

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
st.title("Ask PDFs")

# Design the chatbot
llm = ChatOllama(base_url="http://localhost:11434", model=llm_name, temperature=temperature)
system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant. You explain things in short and brief. If you don't know or unsure about question, say I dont know")

# Design the upload
uploaded_files = st.file_uploader(
    "Choose a PDF file", 
    accept_multiple_files=True,
    type="pdf")

if uploaded_files:
    #print(uploaded_files)
    docs = []
    for uploaded_file in uploaded_files:
        tempfile = "./pdf/temp.pdf"
        with open(tempfile,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        loader = PyPDFLoader(tempfile)
        docs.extend(loader.load())
    #Split
    splits = split_text(docs)
    
    # Embedding
    persist_directory = "./docs/chroma"
    oembeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

    print("Starting to embed")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=oembeddings,
        persist_directory=persist_directory
    )
    # Design Q&A
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    qtext = st.chat_input("What do you want to ask from PDF?")

    if qtext:
        st.write(f"**:adult: User**: {qtext}")
        with st.spinner("Thinking..."):
            response = generate_response(qtext)
            st.write(f"**:horse: Peruna**: {response}")
