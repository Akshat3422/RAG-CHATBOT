from fastapi import FastAPI
from pydantic import BaseModel
import os 
from prompts.llm_prompt import final_prompt
from utils import init_app, join_context
from rag.data_retrieving import retrieve_query
from rag.final_pipeline import ingest_documents, rag_query  # or wherever defined
from fastapi import UploadFile, File


# ---------- App Init ----------
app = FastAPI(title="RAG Chatbot", version="1.0")

app_state = init_app(final_prompt)

index = app_state["index"]
model = app_state["embedding_model"]
chain = app_state["chain"]

DEFAULT_NAMESPACE = "recent_wars_v1"


UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploaded_docs")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# ---------- Request Models ----------
class ChatRequest(BaseModel):
    query: str
    namespace: str | None = DEFAULT_NAMESPACE


class IngestRequest(BaseModel):
    pdf_path: str
    namespace: str

# ---------- Routes ----------

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG chatbot running 🚀"}


@app.post("/ingest")
async def ingest(
    namespace: str,
    file: UploadFile = File(...)
):
    file_path = os.path.join(UPLOAD_DIRECTORY, str(file.filename))

    with open(file_path, "wb") as f:
        f.write(await file.read())

    ingest_documents(
        pdf_path=file_path,
        namespace=namespace
    )

    return {
        "status": "success",
        "file": file.filename,
        "namespace": namespace
    }



@app.post("/chat")
def chat(req: ChatRequest):
    answer = rag_query(
        query=req.query,
        namespace=req.namespace #type: ignore
    ) 
    return {
        "query": req.query,
        "answer": answer
    }
