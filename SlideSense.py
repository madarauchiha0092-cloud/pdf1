import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="SlideSense PDF Analyser",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%); color: #ffffff; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .hero-section { text-align: center; padding: 3rem; margin-bottom: 2rem; }
    .hero-title { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #1e40af); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .success-notification { background: rgba(34,197,94,0.2); border-radius: 12px; padding: 1rem; margin-top: 1rem; }
    .success-text { color: #22c55e; font-weight: 600; }
    .query-section { background: rgba(30,41,59,0.6); border-radius: 12px; padding: 2rem; margin-top: 2rem; }
    .response-section { background: rgba(30,41,59,0.8); border-radius: 12px; padding: 2rem; margin-top: 2rem; }
    .response-title { font-size: 1.25rem; font-weight: 600; color: #f8fafc; }
    .response-content { color: #e2e8f0; font-size: 1.1rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">SlideSense PDF Analyser</h1>
    <p>Advanced Document Analysis with AI Technology</p>
</div>
""", unsafe_allow_html=True)

# File uploader
pdf = st.file_uploader("Upload a PDF document", type="pdf")

if pdf is not None:
    with st.spinner('Processing your document...'):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        splitted_text = text_splitter.split_text(text)

        # Clean chunks (remove empty or None)
        splitted_text = [chunk for chunk in splitted_text if isinstance(chunk, str) and chunk.strip()]

        # Embeddings with safe max_length
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            encode_kwargs={'max_length': 512}
        )

        # Create vector DB
        vector_db = FAISS.from_texts(splitted_text, embeddings)

    st.markdown("""
    <div class="success-notification">
        <p class="success-text">✅ Document processed successfully and ready for queries</p>
    </div>
    """, unsafe_allow_html=True)

    # Query section
    st.markdown("""
    <div class="query-section">
        <h3>Ask intelligent questions about your document</h3>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input("Ask a question", placeholder="Enter your question...")

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
            <h3 class="response-title">Response</h3>
            <div class="response-content">{response}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("📘 Upload a PDF document above to unlock AI-powered analysis.")
