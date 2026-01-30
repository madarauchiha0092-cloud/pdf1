import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="SlideSense",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# -------------------- BLIP Model for Image Recognition --------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image(image: Image.Image):
    """Return text description for uploaded image."""
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# -------------------- Sidebar for Page Selection --------------------
page = st.sidebar.selectbox("Choose Page", ["PDF Analyzer", "Image Recognition"])

# -------------------- CSS Animations & Styling (unchanged) --------------------
st.markdown("""
<style>
/* ALL your previous CSS for hero, glass-card, response-section, uploader, animations */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%); background-attachment: fixed; font-family: 'Inter', sans-serif; color: #ffffff; min-height: 100vh; }
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
* { transition: all 0.4s cubic-bezier(0.4,0,0.2,1);}
.main .block-container { padding:2rem; max-width:1400px; margin:0 auto;}
.hero-section { text-align:center; padding:5rem 2rem; margin-bottom:3rem; background: radial-gradient(ellipse at center, rgba(59,130,246,0.1) 0%, transparent 70%); border-radius:24px; position:relative; overflow:hidden;}
.hero-section::before {content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:linear-gradient(45deg,transparent,rgba(59,130,246,0.03),transparent);animation:rotate 20s linear infinite;pointer-events:none;}
@keyframes rotate {0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
.hero-title {font-size:4.2rem;font-weight:800;letter-spacing:2px;background:linear-gradient(135deg,#60a5fa 0%,#3b82f6 40%,#1e40af 100%);background-clip:text;-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1.5rem;position:relative;z-index:1;text-shadow:0 0 40px rgba(59,130,246,0.5),0 0 80px rgba(59,130,246,0.3);}
.hero-title::after {content:'';display:block;width:120px;height:4px;margin:1rem auto 0 auto;border-radius:4px;background:linear-gradient(90deg,#1e40af,#3b82f6,#60a5fa);background-size:200% 100%;animation:underlineShift 4s ease infinite;}
@keyframes underlineShift {0%{background-position:0% 50%;}100%{background-position:100% 50%;}}
.hero-subtitle {font-size:1.25rem;color:#cbd5e1;font-weight:400;position:relative;z-index:1;opacity:0.9;}
.glass-card {background:rgba(15,23,42,0.6);backdrop-filter:blur(20px);border:1px solid rgba(59,130,246,0.2);border-radius:20px;padding:2.5rem;margin-bottom:2rem;box-shadow:0 8px 32px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.1);position:relative;overflow:hidden;transition:all 0.4s cubic-bezier(0.4,0,0.2,1);}
.glass-card::before {content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(59,130,246,0.1),transparent);transition:all 0.6s ease;}
.glass-card:hover {transform:translateY(-8px);border-color:rgba(59,130,246,0.4);box-shadow:0 20px 40px rgba(0,0,0,0.4),0 0 30px rgba(59,130,246,0.2),inset 0 1px 0 rgba(255,255,255,0.2);}
.glass-card:hover::before {left:100%;}
.query-section {background:linear-gradient(135deg,rgba(15,23,42,0.8) 0%,rgba(30,41,59,0.6) 100%);backdrop-filter:blur(20px);border:1px solid rgba(59,130,246,0.3);border-radius:20px;padding:2.5rem;position:relative;overflow:hidden;margin:2rem 0;}
.query-section::before {content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#3b82f6,transparent);animation:shimmer 3s ease-in-out infinite;}
@keyframes shimmer {0%,100%{opacity:0;}50%{opacity:1;}}
.response-section {background:linear-gradient(135deg,rgba(15,23,42,0.9) 0%,rgba(30,41,59,0.7) 100%);backdrop-filter:blur(25px);border:1px solid rgba(59,130,246,0.4);border-radius:20px;padding:2.5rem;margin:2rem 0;position:relative;overflow:hidden;box-shadow:0 20px 40px rgba(0,0,0,0.3),0 0 20px rgba(59,130,246,0.1);}
.response-section::before {content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1e40af,#3b82f6,#60a5fa,#3b82f6,#1e40af);background-size:300% 100%;animation:gradientShift 4s ease infinite;}
@keyframes gradientShift {0%,100%{background-position:0% 50%;}50%{background-position:100% 50%;}}
.response-header {display:flex;align-items:center;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(59,130,246,0.2);}
.response-icon {font-size:1.5rem;margin-right:0.75rem;color:#3b82f6;}
.response-title {font-size:1.25rem;font-weight:600;color:#f8fafc;margin:0;}
.response-content {color:#e2e8f0;font-size:1.1rem;line-height:1.8;position:relative;z-index:1;white-space:pre-wrap;word-wrap:break-word;}
.response-content p {margin-bottom:1rem;text-align:justify;}
.response-content strong {color:#60a5fa;font-weight:600;}
.response-content em {color:#a78bfa;font-style:italic;}
.response-content ul, .response-content ol {margin:1rem 0;padding-left:2rem;}
.response-content li {margin-bottom:0.5rem;color:#cbd5e1;}
.response-content code {background:rgba(59,130,246,0.1);color:#60a5fa;padding:0.25rem 0.5rem;border-radius:6px;font-family:'Fira Code',monospace;font-size:0.9rem;}
.success-notification {background:linear-gradient(135deg,rgba(34,197,94,0.1) 0%,rgba(16,185,129,0.1) 100%);border:1px solid rgba(34,197,94,0.3);border-radius:12px;padding:1rem 2rem;text-align:center;position:relative;overflow:hidden;animation:slideInUp 0.6s cubic-bezier(0.4,0,0.2,1);margin:2rem 0;}
@keyframes slideInUp {from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
.success-text {color:#22c55e;font-weight:500;margin:0;}
.stTextInput>div>div>input {background:rgba(15,23,42,0.8)!important;color:#e2e8f0!important;border:2px solid rgba(59,130,246,0.3)!important;border-radius:12px!important;padding:1rem 1.5rem!important;font-size:1rem!important;font-family:'Inter',sans-serif!important;backdrop-filter:blur(10px)!important;transition:all 0.4s cubic-bezier(0.4,0,0.2,1)!important;box-shadow:0 4px 12px rgba(0,0,0,0.1)!important;}
.stTextInput>div>div>input:focus {border-color:rgba(59,130,246,0.8)!important;box-shadow:0 0 0 4px rgba(59,130,246,0.1),0 8px 20px rgba(0,0,0,0.2),0 0 30px rgba(59,130,246,0.3);transform:translateY(-2px)!important;outline:none!important;background:rgba(15,23,42,0.9)!important;}
.stTextInput>div>div>input::placeholder {color:#64748b!important;opacity:0.8!important;}
.stFileUploader {background:rgba(15,23,42,0.4);border:2px dashed rgba(59,130,246,0.3);border-radius:20px;padding:3rem 2rem;transition:all 0.4s cubic-bezier(0.4,0,0.2,1);position:relative;overflow:hidden;margin:2rem 0;}
.stFileUploader::before {content:'Document Upload';position:absolute;top:1rem;left:50%;transform:translateX(-50%);color:#60a5fa;font-size:1.5rem;font-weight:700;font-family:'Inter',sans-serif;padding:0.1rem 1rem;text-shadow:0 0 15px rgba(59,130,246,0.3);z-index:2;}
.stFileUploader::after {content:'Select your PDF file to begin intelligent analysis';position:absolute;top:3.5rem;left:50%;transform:translateX(-50%);color:#94a3b8;font-size:1rem;z-index:2;text-align:center;width:100%;}
.stFileUploader:hover {border-color:rgba(59,130,246,0.6);background:rgba(15,23,42,0.6);box-shadow:0 0 40px rgba(59,130,246,0.15);transform:scale(1.02);}
.stFileUploader>div {background:transparent!important;border:none!important;margin-top:3rem!important;}
.stFileUploader label {display:none!important;}
.stFileUploader [data-testid="stFileUploaderDropzone"] {background:transparent!important;border:none!important;color:#e2e8f0!important;text-align:center!important;padding:2rem!important;min-height:100px!important;}
.stFileUploader [data-testid="stFileUploaderDropzone"] div {color:#cbd5e1!important;font-size:1rem!important;}
.section-title {font-size:1.5rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;text-align:center;position:relative;}
.section-subtitle {color:#94a3b8;text-align:center;margin-bottom:2rem;font-size:1rem;}
.stSpinner>div {border-top-color:#3b82f6!important;animation:spin 1s linear infinite!important;}
.instruction-container {text-align:center;padding:4rem 2rem;background:rgba(15,23,42,0.3);border-radius:20px;border:1px solid rgba(59,130,246,0.2);backdrop-filter:blur(10px);}
.instruction-container:hover {transform:translateY(-8px);border-color:rgba(59,130,246,0.5);box-shadow:0 20px 40px rgba(0,0,0,0.4),0 0 30px rgba(59,130,246,0.3),inset 0 1px 0 rgba(255,255,255,0.2);background:rgba(15,23,42,0.5);}
.instruction-title {font-size:2rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;}
.instruction-text {font-size:1.1rem;color:#cbd5e1;line-height:1.7;max-width:600px;margin:0 auto;}
@media(max-width:768px){.hero-title{font-size:2.5rem}.glass-card{padding:2rem}.response-section{padding:2rem}.response-content{font-size:1rem}}
</style>
""", unsafe_allow_html=True)

# -------------------- HERO SECTION --------------------
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">SlideSense</h1>
    <p class="hero-subtitle">Advanced Document & Image Analysis with AI Technology</p>
</div>
""", unsafe_allow_html=True)

# -------------------- PDF Analyzer --------------------
if page == "PDF Analyzer":
    pdf = st.file_uploader("Document Upload", type="pdf", help="Choose a PDF document for analysis")
    
    if pdf is not None:
        with st.spinner('Processing your document...'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page_pdf in pdf_reader.pages:
                text += page_pdf.extract_text()
                text += "\n\n"

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

        user_query = st.text_input(
            "Ask a question",
            placeholder="Enter your question about the document...",
            help="Type your question here",
            label_visibility="collapsed"
        )

        if user_query:
            with st.spinner('🤖 Generating intelligent response...'):
                docs = vector_db.similarity_search(user_query)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                prompt = ChatPromptTemplate.from_template(
                    "Answer the following:\n{context}\nQuestion: {question}\n Read the context carefully and then answer it"
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

# -------------------- IMAGE RECOGNITION --------------------
if page == "Image Recognition":
    image_file = st.file_uploader("Upload an Image", type=["png","jpg","jpeg"], help="Choose an image for recognition")
    
    if image_file is not None:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            description = describe_image(img)
        
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
            <div class="instruction-title">Image
