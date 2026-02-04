import re
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser


# Loading the entire environments variable
load_dotenv()


pinecone_api_key=os.getenv('PINECONE_API_KEY')
groq_api_key=os.getenv("GROQ_API_KEY")



def clean_text_hard(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "\u25A0": " ",   # black square ■
        "\u00a0": " ",
        "\u202f": " ",
        "\u2009": " ",
        "\u2007": " ",
        "\u200b": "",
        "\u2060": "",
        "–": "-",
        "—": "-",
        "‐": "-",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Fix broken words like counter offensives
    text = re.sub(r"\s+", " ", text)

    return text.strip()





def join_context(matches):
    texts = []

    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text")
        if text:
            texts.append(text)

    return "\n\n".join(texts)




def init_app(final_prompt):
    # Initialize LLM
    llm = ChatGroq(
        api_key=groq_api_key,# type: ignore
        model="openai/gpt-oss-120b"  # type: ignore
    )

    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize chain
    chain = final_prompt | llm | StrOutputParser()

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(
        name="vectordb",
        pool_threads=50,
        connection_pool_maxsize=50
    )

    return {
        "llm": llm,
        "embedding_model": embedding_model,
        "chain": chain,
        "index": index
    }
