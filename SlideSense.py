import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import httpx

# Page configuration
st.set_page_config(
    page_title="SlideSense PDF Analyser", 
    page_icon="📘", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# Ultra-smooth dark blue & black themed CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%); background-attachment: fixed; font-family: 'Inter', sans-serif; color: #ffffff; min-height: 100vh; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stDeployButton {visibility: hidden;}
    * { transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
    .main .block-container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
    .hero-section { text-align: center; padding: 5rem 2rem; margin-bottom: 3rem; background: radial-gradient(ellipse at center, rgba(59, 130, 246, 0.1) 0%, transparent 70%); border-radius: 24px; position: relative; overflow: hidden; }
    .hero-title { font-size: 4.2rem !important; font-weight: 800; letter-spacing: 2px; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 40%, #1e40af 100%); background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem; }
    .glass-card { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(20px); border-radius: 20px; padding: 2.5rem; margin-bottom: 2rem; }
    .query-section { background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%); backdrop-filter: blur(20px); border-radius: 20px; padding: 2.5rem; position: relative; overflow: hidden; margin: 2rem 0; }
    .response-section { background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); backdrop-filter: blur(25px); border-radius: 20px; padding: 2.5rem; margin: 2rem 0; position: relative; }
    .response-header { display: flex; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; }
    .response-title { font-size: 1.25rem; font-weight: 600; color: #f8fafc; margin: 0; }
    .response-content { color: #e2e8f0; font-size: 1.1rem; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word; }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">SlideSense PDF Analyser</h1>
    <p class="hero-subtitle">Advanced Document Analysis with AI Technology</p>
</div>
""", unsafe_allow_html=True)

# File uploader
pdf = st.file_uploader("Document Upload", type="pdf", help="Choose a PDF document for analysis")

if pdf is not None:
    # Process PDF
    with st.spinner('Processing your document...'):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        splitted_text = text_splitter.split_text(text)

        # Set a longer timeout for Hugging Face model
        client = httpx.Client(timeout=httpx.Timeout(30.0))  # 30 seconds timeout

        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            client=client  # pass client here
        )

        # Create vector DB
        vector_db = FAISS.from_texts(splitted_text, embeddings)

    # Success message
    st.markdown("""
    <div class="success-notification">
        <p class="success-text">✅ Document processed successfully and ready for queries</p>
    </div>
    """, unsafe_allow_html=True)

    # Query section
    st.markdown("""
    <div class="query-section">
        <div class="section-title">Query Interface</div>
        <div class="section-subtitle">Ask intelligent questions about your document</div>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "Ask a question",  # Non-empty label
        placeholder="Enter your question about the document...",
        help="Type your question here",
        label_visibility="collapsed"
    )

    if user_query:
        with st.spinner('🤖 Generating intelligent response...'):
            docs = vector_db.similarity_search(user_query)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            prompt = ChatPromptTemplate.from_template("Answer the following:\n{context}\nQuestion: {question}\n Read the context carefully and then answer it")
            chain = create_stuff_documents_chain(llm, prompt)  
            response = chain.invoke({"context": docs, "question": user_query})

        # Display response with enhanced formatting
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
    # Instructions
    st.markdown("""
    <div class="instruction-container">
        <div class="instruction-title">Get Started</div>
        <div class="instruction-text">
            Upload your PDF document above to unlock the power of AI-driven document analysis. 
            Ask questions, extract insights, and discover information with intelligent search capabilities.
        </div>
    </div>
    """, unsafe_allow_html=True)
