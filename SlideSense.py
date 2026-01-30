import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from PIL import Image
import asyncio
import google.generativeai as genai
import os

# Page configuration
st.set_page_config(
    page_title="SlideSense",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()
genai.api_key = os.getenv("GOOGLE_API_KEY")

# --- CSS Styling (unchanged) ---
st.markdown("""<style>
/* (CSS from previous code here) */
</style>""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">SlideSense</h1>
    <p class="hero-subtitle">Advanced Document & Image Analysis with AI</p>
</div>
""", unsafe_allow_html=True)

# --- Tabs for PDF Analyzer & Image Recognition ---
tab1, tab2 = st.tabs(["📘 PDF Analyzer", "🖼️ Image Recognition"])

# -----------------------------
# PDF Analyzer Tab
# -----------------------------
with tab1:
    pdf = st.file_uploader("Document Upload", type="pdf", help="Choose a PDF document for analysis")
    if pdf is not None:
        with st.spinner('Processing your document...'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            splitted_text = text_splitter.split_text(text)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            vector_db = FAISS.from_texts(splitted_text, embeddings)

        st.markdown("""
        <div class="success-notification">
            <p class="success-text">✅ Document processed successfully and ready for queries</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="query-section">
            <div class="section-title">Query Interface</div>
            <div class="section-subtitle">Ask intelligent questions about your document</div>
        </div>
        """, unsafe_allow_html=True)

        user_query = st.text_input("Ask a question", placeholder="Enter your question about the document...", label_visibility="collapsed")

        if user_query:
            with st.spinner('🤖 Generating intelligent response...'):
                docs = vector_db.similarity_search(user_query)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                prompt = ChatPromptTemplate.from_template(
                    "Answer the following:\n{context}\nQuestion: {question}\nRead the context carefully and then answer it"
                )
                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({"context": docs, "question": user_query})

            st.markdown(f"""
            <div class="response-section">
                <div class="response-header">
                    <span class="response-icon">🤖</span>
                    <h3 class="response-title">Response</h3>
                </div>
                <div class="response-content">{response}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="instruction-container">
            <div class="instruction-title">Get Started</div>
            <div class="instruction-text">
                Upload your PDF document above to unlock the power of AI-driven document analysis.
                Ask questions, extract insights, and discover information with intelligent search capabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Image Recognition Tab
# -----------------------------
with tab2:
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"], help="Upload an image for AI recognition")
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, use_container_width=True)

        with st.spinner("🔍 Recognizing image..."):
            try:
                description_response = genai.Image.generate_descriptions(image=image_file)
                description = description_response[0].caption
            except Exception as e:
                description = f"⚠️ Could not generate description: {str(e)}"

        st.markdown(f"""
        <div class="response-section">
            <div class="response-header">
                <span class="response-icon">🖼️</span>
                <h3 class="response-title">Image Description</h3>
            </div>
            <div class="response-content">{description}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="instruction-container">
            <div class="instruction-title">Get Started with Image Recognition</div>
            <div class="instruction-text">
                Upload an image above to receive an AI-generated description of the visual content.
            </div>
        </div>
        """, unsafe_allow_html=True)
