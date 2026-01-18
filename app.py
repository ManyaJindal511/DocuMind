import streamlit as st
from PyPDF2 import PdfReader
from pathlib import Path
import shutil
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# ---------------- EMBEDDINGS (FIXED) ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}   # ✅ CRITICAL FIX
    )

# ---------------- PDF TEXT EXTRACTION ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# ---------------- CLEAN PDF TEXT ----------------
def clean_text(text):
    # Fix broken hyphenated words
    text = re.sub(r"-\s+", "", text)

    # Remove citation numbers like 90, 91, 92
    text = re.sub(r"\b\d{2,}\b", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------- TEXT CHUNKING ----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# ---------------- VECTOR STORE ----------------
def get_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

# ---------------- FREE HF LLM ----------------
@st.cache_resource
def get_llm():
    model_id = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3
    )

    return HuggingFacePipeline(pipeline=pipe)

# ---------------- RAG CHAIN ----------------
def get_rag_chain():
    llm = get_llm()

    prompt = PromptTemplate(
        template="""
Answer the question using ONLY the context below.
If the answer is not present in the context, say:
"The answer is not present in the document."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    return prompt | llm | StrOutputParser()

# ---------------- USER QUERY ----------------
def user_input(question):
    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = get_rag_chain()
    response = chain.invoke({
        "context": context,
        "question": question
    })

    st.write("### ✅ Answer")
    st.write(response)

# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
    st.header("📄 RAG PDF Chatbot")

    index_exists = Path("faiss_index").exists()

    user_question = st.text_input(
        "Ask a question from the PDF",
        disabled=not index_exists
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload one or more PDFs",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF")
                return

            if Path("faiss_index").exists():
                shutil.rmtree("faiss_index")

            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                cleaned_text = clean_text(raw_text)
                chunks = get_text_chunks(cleaned_text)
                get_vector_store(chunks)

            st.success("✅ PDFs processed successfully! You can now ask questions.")

if __name__ == "__main__":
    main()
