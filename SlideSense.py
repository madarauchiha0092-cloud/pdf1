import os
import streamlit as st
import asyncio
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

import google.generativeai as genai


# -------------------- APP CONFIG --------------------
st.set_page_config(
    page_title="SlideSense PDF Analyser",
    page_icon="📘",
    layout="wide"
)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Set it in .env or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Fix asyncio loop issue (important for Streamlit)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# -------------------- IMAGE RECOGNITION FUNCTION --------------------
def describe_image(image: Image.Image) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            "Describe this image clearly and simply for a project demonstration.",
            image
        ]
    )
    return response.text


# -------------------- UI HEADER --------------------
st.markdown("""
<h1 style="text-align:center;">📘 SlideSense PDF Analyser</h1>
<p style="text-align:center;">PDF Question Answering + Image Recognition Extension</p>
<hr>
""", unsafe_allow_html=True)


# -------------------- PDF UPLOAD --------------------
pdf = st.file_uploader("Upload PDF Document", type="pdf")

if pdf:
    with st.spinner("📄 Processing PDF..."):
        reader = PdfReader(pdf)
        full_text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )

        chunks = splitter.split_text(full_text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_texts(chunks, embeddings)

    st.success("✅ PDF processed successfully")

    user_query = st.text_input("Ask a question about the PDF")

    if user_query:
        with st.spinner("🤖 Generating answer..."):
            docs = vector_db.similarity_search(user_query)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )

            prompt = ChatPromptTemplate.from_template(
                "Answer the question using only the context below:\n\n{context}\n\nQuestion: {question}"
            )

            chain = create_stuff_documents_chain(llm, prompt)

            # ✅ FIXED: chain returns STRING, not dict
            answer = chain.invoke({
                "context": docs,
                "question": user_query
            })

        st.markdown("### 🤖 Answer")
        st.write(answer)


# -------------------- IMAGE RECOGNITION EXTENSION --------------------
st.markdown("---")
st.markdown("## 🖼️ Image Recognition Extension")

image_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["png", "jpg", "jpeg"]
)

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Recognizing image..."):
        description = describe_image(image)

    st.markdown("### 🧠 Image Description")
    st.write(description)


# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px;">
SlideSense Project • PDF Analysis + Image Recognition
</p>
""", unsafe_allow_html=True)
