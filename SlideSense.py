import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# OCR imports
from pdf2image import convert_from_bytes
import pytesseract

st.set_page_config(page_title="SlideSense PDF Analyser", page_icon="📘", layout="wide")
load_dotenv()

st.title("📘 SlideSense PDF Analyser")

pdf = st.file_uploader("Upload a PDF document", type="pdf")

if pdf is not None:
    with st.spinner("Processing your document..."):
        text = ""

        # Try normal text extraction first
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

        # If no text found, fallback to OCR
        if not text.strip():
            st.warning("No text detected with PyPDF2. Using OCR fallback...")
            images = convert_from_bytes(pdf.read())
            for img in images:
                text += pytesseract.image_to_string(img) + "\n\n"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        splitted_text = text_splitter.split_text(text)
        splitted_text = [chunk for chunk in splitted_text if isinstance(chunk, str) and chunk.strip()]

        if not splitted_text:
            st.error("❌ No valid text could be extracted from the PDF.")
            vector_db = None
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"max_length": 512, "truncation": True}
            )
            vector_db = FAISS.from_texts(splitted_text, embeddings)

    if vector_db:
        st.success("✅ Document processed successfully and ready for queries")

        user_query = st.text_input("Ask a question about the document")
        if user_query:
            with st.spinner("🤖 Generating intelligent response..."):
                docs = vector_db.similarity_search(user_query)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                prompt = ChatPromptTemplate.from_template(
                    "Answer the following:\n{context}\nQuestion: {question}\nRead the context carefully and then answer it"
                )
                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({"context": docs, "question": user_query})

            st.markdown(f"### Response\n{response}")
else:
    st.info("📘 Upload a PDF document above to unlock AI-powered analysis.")
