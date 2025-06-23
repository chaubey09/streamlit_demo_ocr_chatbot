# Final, Streamlit-Cloud-Compatible Version
# Removed PySpark; replaced with Pandas
# Disabled Keycloak login logic for public access

import base64
import pytesseract
from PIL import Image
import pandas as pd
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import streamlit as st
import os

# --- UI Setup ---
st.set_page_config(page_title="AI Chat App with OCR", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .bio-card { 
        background-color: #ffffff; 
        border-radius: 12px; 
        padding: 16px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        text-align: center; 
        margin-bottom: 20px; 
    }
    .bio-card img { 
        border-radius: 50%; 
        width: 150px !important; 
        height: 150px; 
        object-fit: cover; 
    }
    .bio-card a { 
        color: #4a90e2; 
        text-decoration: none; 
    }
    .bio-card a:hover { 
        text-decoration: underline; 
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Chat Application with OCR, FAISS, and Gemini")
st.success("🧪 Public demo mode enabled (authentication disabled)")

# Sidebar
with st.sidebar:
    use_case = st.selectbox("Choose a Use Case", ["OCR Extraction", "Summarization", "Question and Answer", "Text Generation"])
    st.subheader("Contact Me!")
    try:
        profile_pic = Image.open("IMG_0924(1).jpg")
        st.image(profile_pic, width=150, caption="Anmol Chaubey", output_format="PNG")
    except FileNotFoundError:
        st.warning("Profile photo not found.")
    st.markdown("""
        <div class="bio-card">
            <strong>Anmol Chaubey</strong><br>
            Email: <a href="mailto:anmolchaubey820@gmail.com">anmolchaubey820@gmail.com</a><br>
            <a href="https://www.linkedin.com/in/anmol-chaubey-120b42206/" target="_blank">LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)

# --- Load Data and Create FAISS Index ---
@st.cache_data
def load_data_and_create_index():
    df = pd.read_csv("healthcare_dataset.csv").head(10000)
    documents = df.to_dict(orient="records")
    text_documents = [" ".join(str(v) for v in doc.values()) for doc in documents]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store.as_retriever()

retriever = load_data_and_create_index()

# Gemini API Key (set this in Streamlit Cloud secrets)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Use Case Logic ---
if use_case == "OCR Extraction":
    st.header("📄 OCR Text Extraction")
    uploaded_file = st.file_uploader("Upload an image for OCR", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text_area("Extracted Text:", text, height=200)

elif use_case == "Summarization":
    st.header("📝 Summarization")
    query = st.text_input("Enter a search query:")
    if st.button("Summarize") and query.strip():
        docs = retriever.get_relevant_documents(query)
        if docs:
            content = " ".join([doc.page_content for doc in docs])
            response = llm.generate_content(f"Summarize this: {content}")
            st.write(response.text)
        else:
            st.warning("No relevant documents found.")

elif use_case == "Question and Answer":
    st.header("💬 Question and Answer")
    question = st.text_input("Ask a question:")
    if st.button("Get Answer") and question.strip():
        docs = retriever.get_relevant_documents(question)
        if docs:
            context = " ".join([doc.page_content for doc in docs])
            response = llm.generate_content(f"Answer this based on context: {question}\nContext: {context}")
            st.write(response.text)
        else:
            st.warning("No relevant documents found.")

elif use_case == "Text Generation":
    st.header("🖊 Text Generation")
    prompt = st.text_area("Enter a prompt:")
    if st.button("Generate Text") and prompt.strip():
        response = llm.generate_content(prompt)
        st.write(response.text)
