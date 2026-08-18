# 🏗️ DESIGNCHECK — Zoning & Building Code Compliance AI Agent

An intelligent, RAG-powered compliance engine designed to streamline architectural plan reviews and real estate due diligence. **DESIGNCHECK** parses municipal zoning codes, building regulations, and land-use ordinances from uploaded PDFs, allowing architects, urban planners, and developers to instantly audit proposed project parameters against official codes.

---

## 🌟 Key Features

* **Automated PDF Parsing & Chunking:** Ingests complex zoning documents and chunks text using `RecursiveCharacterTextSplitter`.
* **Fast Local Vector Search:** Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and `ChromaDB` for instant, CPU-friendly vector indexing without API embedding costs.
* **Structured Regulatory Extraction:** Leverages Google's `Gemini 3.6 Flash` to deliver structured compliance outputs:
  * **Status:** Compliant / Non-Compliant / Needs Review
  * **Rule Summary:** Brief breakdown of statutory requirements
  * **Exact Citation / Section:** Verifiable document citations
* **Transparent Citations:** Provides expandable document snippet references with exact page numbers for auditability.
* **Streamlit Web UI:** Clean, intuitive interface for uploading documents and running automated queries.

---

## 🛠️ Tech Stack

* **Framework:** [LangChain](https://www.langchain.com/) (LCEL Pipeline)
* **LLM:** Google Gemini (`gemini-3.6-flash`) via `langchain-google-genai`
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` via `langchain-huggingface`
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Frontend/App Framework:** [Streamlit](https://streamlit.io/)
* **Language:** Python 3.10+

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher installed.
* A Google AI Studio API Key ([Get a Gemini API Key](https://aistudio.google.com/)).

### Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/okikiolalb/DESIGNCHECK.git](https://github.com/okikiolalb/DESIGNCHECK.git)
   cd DESIGNCHECK
