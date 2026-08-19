from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
import io
import json
import os
import re
import tempfile
import PIL.Image
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Verify API key
gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not gemini_key:
    st.error("⚠️ GEMINI_API_KEY not found. Please check your .env file.")

# LangChain Imports

# 1. Setup & Config
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

st.title("🏗️ DESIGNCHECK — Multimodal Compliance Agent")

# 2. Initialize Models


@st.cache_resource
def get_models():
    # Vision & Text LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=gemini_key
    )
    # Local CPU Embeddings for PDFs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings


llm, embeddings = get_models()

# 3. Sidebar Uploads
st.sidebar.header("Project Documents")
uploaded_code_pdf = st.sidebar.file_uploader(
    "1. Upload Zoning Code (PDF)", type=["pdf"])
uploaded_plan_img = st.sidebar.file_uploader(
    "2. Upload Architectural Plan (Image)", type=["png", "jpg", "jpeg"])

# Helper function to format retrieved context documents into a clean string


def format_docs(docs):
    return "\n\n".join(f"[Source Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}" for doc in docs)


rag_chain = None

# 4. Process Zoning Code (RAG Pipeline)
if uploaded_code_pdf:
    with st.spinner("Indexing Zoning Code PDF..."):
        # Save temp PDF to allow LangChain to read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_code_pdf.getvalue())
            tmp_pdf_path = tmp_file.name

        # Load and Chunk the PDF
        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="zoning_code"
        )

        # DEFINE RETRIEVER HERE
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Define RAG Prompt
        system_prompt = (
            "You are an expert municipal plan reviewer. Use the following context from the zoning code to answer. "
            "If the answer is not in the context, say so. Always cite section numbers or page references.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Build Modern LCEL RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.sidebar.success("✅ Zoning Code Indexed!")

# 5. Process Plan Drawing (Vision Pipeline) & Run Audit
if uploaded_plan_img:
    image = PIL.Image.open(uploaded_plan_img).convert("RGB")
    st.image(image, caption="Uploaded Plan Drawing", use_container_width=True)

    if st.button("Run Multimodal Audit"):
        if not uploaded_code_pdf or rag_chain is None:
            st.error(
                "⚠️ Please upload a Zoning Code (PDF) first so I have rules to check against!")
        else:
            with st.spinner("Extracting spatial metrics from architectural plan..."):
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data_uri = f"data:image/jpeg;base64,{img_str}"

                vision_prompt = """
                Extract the following spatial metrics from this architectural plan.
                Return ONLY a valid raw JSON object. Do NOT wrap it in markdown code blocks or additional text.

                {
                    "proposed_building_height_meters": float or null,
                    "proposed_stories": int or null,
                    "proposed_lot_coverage_percentage": float or null,
                    "proposed_front_setback_meters": float or null,
                    "proposed_rear_setback_meters": float or null,
                    "proposed_side_setback_meters": float or null,
                    "provided_parking_spaces": int or null
                }
                """

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                )

                raw_response = llm.invoke([message])

                # Unpack content string if returned inside a list/dict payload
                content = raw_response.content
                if isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if isinstance(
                        item, dict) and "text" in item]
                    raw_text = "".join(text_parts)
                else:
                    raw_text = str(content)

                # Extract the JSON substring bounded by curly braces
                try:
                    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        extracted_data = json.loads(json_str)
                    else:
                        raise ValueError(
                            "No JSON block enclosed in braces found in response.")

                    st.subheader("📐 Extracted Plan Parameters")
                    st.json(extracted_data)

                    with st.spinner("Auditing extracted parameters against Zoning Code..."):
                        rag_query = f"""
                        Audit the following proposed building parameters against the zoning code:
                        - Height: {extracted_data.get('proposed_building_height_meters')} meters ({extracted_data.get('proposed_stories')} stories)
                        - Lot Coverage: {extracted_data.get('proposed_lot_coverage_percentage')}%
                        - Front Setback: {extracted_data.get('proposed_front_setback_meters')} meters
                        - Rear Setback: {extracted_data.get('proposed_rear_setback_meters')} meters
                        - Side Setback: {extracted_data.get('proposed_side_setback_meters')} meters
                        - Parking Spaces: {extracted_data.get('provided_parking_spaces')}

                        Determine compliance status for each item (Compliant / Non-Compliant / Needs Review) and provide exact section citations.
                        """

                        audit_response = rag_chain.invoke(rag_query)

                        st.subheader("⚖️ Compliance Audit Results")
                        st.markdown(audit_response)

                except Exception as e:
                    st.error(f"Failed to parse extracted metrics: {e}")
                    st.write("Raw Text Evaluated:", raw_text)
