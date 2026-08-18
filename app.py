import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Ensure Gemini API Key is available
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error("Missing GEMINI_API_KEY in .env file.")
    st.stop()

st.set_page_config(page_title="Zoning & Building Code AI Agent",
                   page_icon="🏗️", layout="wide")

st.title("🏗️ Zoning & Building Code Compliance AI Agent")
st.caption(
    "Upload a municipal zoning/building code PDF to evaluate design parameter compliance.")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Zoning Code (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF and generating vector embeddings..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Vector Store using free, reliable local CPU embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Cleanup temp file
        os.remove(tmp_path)

    st.sidebar.success("PDF Indexed Successfully!")

    # Prompt Template for Structured Regulatory Extraction
    system_prompt = (
        "You are an expert AI Architectural Compliance Auditor.\n"
        "Use the following pieces of retrieved zoning context to answer the compliance query.\n"
        "Provide your evaluation in a clear, structured format:\n"
        "- Status: (Compliant / Non-Compliant / Needs Review)\n"
        "- Rule Summary: Brief breakdown of the requirement\n"
        "- Exact Citation / Section: Relevant section from the document\n"
        "If you do not know the answer based on the document, state clearly that the code does not specify.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LLM Setup using Gemini Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=gemini_key
    )

    # LCEL Pipeline Construction
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User Query Input
    user_query = st.text_input(
        "Enter proposed project constraints (e.g., 'Does a 4-story residential building with 60% lot coverage comply?')"
    )

    if user_query:
        with st.spinner("Analyzing against regulatory criteria..."):
            response_text = rag_chain.invoke(user_query)
            retrieved_docs = retriever.invoke(user_query)

            st.subheader("📋 Compliance Audit Result")
            st.write(response_text)

            with st.expander("🔍 Retrieved Code Snippets (Citations)"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(
                        f"**Snippet {idx + 1} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.info(doc.page_content)
else:
    st.info("👈 Please upload a Zoning Code PDF in the sidebar to begin.")
