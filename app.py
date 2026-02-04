import streamlit as st
import requests

# ---------------- Config ----------------
API_BASE_URL = "http://localhost:8000"  # change if deployed
INGEST_URL = f"{API_BASE_URL}/ingest"
CHAT_URL = f"{API_BASE_URL}/chat"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("Upload PDFs & chat using Retrieval-Augmented Generation")

# ---------------- Sidebar: PDF Ingestion ----------------
st.sidebar.header("📄 Ingest Documents")

namespace = st.sidebar.text_input(
    "Namespace",
    value="recent_wars_v1",
    help="Same namespace will be used for retrieval"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if st.sidebar.button("Ingest PDF"):
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF file")
    else:
        with st.spinner("Ingesting document..."): #type: ignore
            files = { 
                "file": (uploaded_file.name, uploaded_file, "application/pdf")
            }
            params = {"namespace": namespace}

            response = requests.post(INGEST_URL, params=params, files=files)

            if response.status_code == 200:
                st.sidebar.success("✅ Document ingested successfully")
                st.sidebar.json(response.json())
            else:
                st.sidebar.error("❌ Ingestion failed")
                st.sidebar.text(response.text)

# ---------------- Main: Chat Section ----------------
st.divider()
st.subheader("💬 Chat with your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question")

if st.button("Send"):
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        payload = {
            "query": query,
            "namespace": namespace
        }

        with st.spinner("Thinking..."):
            response = requests.post(CHAT_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")

            st.session_state.chat_history.append(
                {"query": query, "answer": answer}
            )
        else:
            st.error("❌ Chat request failed")
            st.text(response.text)

# ---------------- Chat History ----------------
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {chat['query']}")
    st.markdown(f"**🤖 Bot:** {chat['answer']}")
    st.markdown("---")
