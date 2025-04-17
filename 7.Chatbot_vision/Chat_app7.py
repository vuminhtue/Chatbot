from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate

import streamlit as st
from PIL import Image
import ollama
import base64

# Define functions
def generate_response(qtext):
    response = qa_chain.run(qtext)
    return response

st.sidebar.title("Settings")
llm_name = st.sidebar.selectbox("Model", ["llama3.2-vision:11b","llama3.2-vision:90b","llava:34b"])
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.5,0.01)
st.sidebar.image("./pony.jpeg")
st.title("Ask Your Image")

# Design the chatbot
llm = ChatOllama(base_url="http://localhost:11434", model=llm_name, temperature=temperature)
system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant. You explain things in short and brief. If you don't know or unsure about question, say I dont know")

# Design the upload

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Read and encode the image in base64
    image_bytes = uploaded_file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Send to ollama
    with st.spinner("Uploading image and analyzing..."):
        try:
            response = ollama.chat(
                model="ingu627/Qwen2.5-VL-7B-Instruct-Q5_K_M:latest",
                messages=[{
                    "role": "user",
                    "content": "Tell me what food, fruit or drink are there in the image and estimate the calories for it?",
                    "images": [image_base64]
                }]
            )
            st.success("Response received!")
            st.write(response["message"]["content"])
        except Exception as e:
            st.error(f"Error: {e}")
