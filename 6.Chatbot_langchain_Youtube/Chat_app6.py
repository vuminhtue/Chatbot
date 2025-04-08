from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma 
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
import yt_dlp
import whisper
import os
import streamlit as st
from pathlib import Path

# Define functions
def generate_response(qtext):
    response = qa_chain.run(qtext)
    return response

def split_text(doc):
    text_splitter = CharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 150,
            add_start_index=True
        )
    return text_splitter.split_text(doc)

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': 'audio/%(title)s.%(ext)s'
    }
 
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Design button
def button1_action():
    st.session_state.button1_clicked = not st.session_state.button1_clicked

def button2_action():
    st.session_state.button2_clicked = not st.session_state.button2_clicked

# Initialize session state variables if not already set
if "button1_clicked" not in st.session_state:
    st.session_state.button1_clicked = False

if "button2_clicked" not in st.session_state:
    st.session_state.button2_clicked = False
    st.session_state.button1_clicked = True


# Design SideBar including title, information on llm model, temperature and images
st.sidebar.title("Settings")
llm_name = st.sidebar.selectbox("Model", ["llama3.2:1b","gemma3:1b","deepseek-r1:1.5b"])
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.5,0.01)
st.sidebar.image("./pony.jpeg")
st.title("Query with Youtube")

# Design the chatbot
llm = ChatOllama(base_url="http://localhost:11434", model=llm_name, temperature=temperature)
system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant. You explain things in short and brief. If you don't know or unsure about question, say I dont know")
model = whisper.load_model("base")


# Design the upload
uploadlink = st.text_area("Upload a Youtube link")
button1 = st.button("Download and Transcribe",on_click=button1_action)
if uploadlink and button1:
    url = uploadlink
    st.video(url)
    with st.spinner("Download and Transcribing..."):
        download_audio(url)
        files = os.listdir("audio")
        for f in os.listdir("audio"):
            result = model.transcribe("audio/"+files[0])
            st.write(result['text'])
            output = "txt/"+f+".txt"
            Path(output).write_text(result['text'])
            os.remove("audio/"+f)

            splits = split_text(result['text'])
            documents = [Document(page_content=text) for text in splits]
            # Embedding
            persist_directory = "./docs/chroma"
            oembeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")
            print("Starting to embed")

            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=oembeddings,
                persist_directory=persist_directory
            )
            st.write("File saved to txt folder!")
            # Design Q&A
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectordb.as_retriever()
            )
            qtext = st.text_area("What do you want to ask from youtube?")
            button2 = st.button("Answer:",on_click=[button1_action,button2_action])
            if qtext and button2:
                st.write(f"**:adult: User**: {qtext}")
                with st.spinner("Thinking..."):
                    response = generate_response(qtext)
                    st.write(f"**:horse: Peruna**: {response}")







