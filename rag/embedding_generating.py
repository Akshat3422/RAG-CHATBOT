from typing import List




def generate_embeddings(model,final_docs:List[str]):
    embeddings = model.encode(final_docs)
    return embeddings


def insertion_in_pinecone(final_docs, embeddings, namespace):
    vectors = []
    for i, (doc, emb) in enumerate(zip(final_docs, embeddings)):
        vectors.append(
            (
                f"{namespace}_{i}",
                emb.tolist(),
                {"text": doc}
            )
        )
    return vectors





# index will be given by init_pinecone function  and insertion of vectors will be given by above function 
def batch_upsert(index, vectors, batch_size=100,namespace=None):
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size],
                                 namespace=namespace )