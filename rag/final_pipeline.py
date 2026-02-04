from rag.data_chunking import document_loader,splitting_the_text,formatting
from rag.embedding_generating import generate_embeddings,insertion_in_pinecone,batch_upsert
from rag.data_retrieving import retrieve_query
from prompts.llm_prompt import final_prompt

from utils import join_context,init_app


app_state = init_app(final_prompt)
index=app_state['index']
model = app_state["embedding_model"]
llm=app_state['llm']
chain=app_state['chain']

def ingest_documents(
    pdf_path: str,
    namespace: str,

    batch_size: int = 100,

):
    docs = document_loader(pdf_path)

    final_chunk_docs = splitting_the_text(docs=docs)

    final_docs = formatting(final_chunk_docs=final_chunk_docs)

    embeddings = generate_embeddings(
        model=model,
        final_docs=final_docs
    )

    vectors = insertion_in_pinecone(
        final_docs=final_docs,
        embeddings=embeddings,
        namespace=namespace
    )

    batch_upsert(
        index=index,
        vectors=vectors,
        batch_size=batch_size,
        namespace=namespace
    )



def rag_query(
    query: str,
    namespace: str,
    chain=chain,
):
    retrieved = retrieve_query(
        index=index,
        model=model,
        query=query,
        namespace=namespace
    )
    if not retrieved:
        return "I don't have enough information in the provided documents to answer this."


    context = join_context(retrieved)

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response
