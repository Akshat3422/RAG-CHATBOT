

def retrieve_query(index,model,query:str,namespace:str):
    vectors=model.encode([query])[0]
    res = index.query(
    vector=vectors.tolist(),
    top_k=50,
    include_metadata=True,
    namespace=namespace
)
    return res['matches'] #type: ignore




