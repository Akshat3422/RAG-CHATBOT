from rag.data_chunking import document_loader,splitting_the_text,formatting
from rag.embedding_generating import generate_embeddings,insertion_in_pinecone,batch_upsert
from rag.data_retrieving import retrieve_query
from prompts.llm_prompt import final_prompt

from utils import join_context,init_app
import bm25s

app_state = init_app(final_prompt)
index = app_state['index']
model = app_state["embedding_model"]
llm = app_state['llm']
chain = app_state['chain']
reranker = app_state['reranker']
bm25_retrievers = app_state['bm25_retrievers']
bm25_corpora = app_state['bm25_corpora']

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

    # BM25 indexing
    if namespace in bm25_corpora:
        bm25_corpora[namespace].extend(final_docs)
    else:
        bm25_corpora[namespace] = final_docs

    corpus_tokens = bm25s.tokenize(bm25_corpora[namespace], stopwords="en")
    text_retriever = bm25s.BM25()
    text_retriever.index(corpus_tokens)
    bm25_retrievers[namespace] = text_retriever


def rag_query(
    query: str,
    namespace: str,
    chain=chain,
):
    # 1. Dense retrieval
    retrieved = retrieve_query(
        index=index,
        model=model,
        query=query,
        namespace=namespace
    )
    dense_docs = join_context(retrieved) if retrieved else []

    # 2. BM25 retrieval
    bm25_docs = []
    if namespace in bm25_retrievers and namespace in bm25_corpora:
        retriever = bm25_retrievers[namespace]
        corpus = bm25_corpora[namespace]
        
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=10)
        
        # Results is usually shape (1, k)
        bm25_docs = [text for text in results[0] if text.strip()]

    # Combine and deduplicate
    combined_docs = dense_docs + bm25_docs
    combined_docs = list(dict.fromkeys(combined_docs))

    if not combined_docs:
        return "I don't have enough information in the provided documents to answer this."

    # 3. Reranking
    pairs = [(query, doc) for doc in combined_docs]
    scores = reranker.predict(pairs)
    
    ranked_docs = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in ranked_docs[:5]]

    context = "\n\n".join(top_docs)

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response
