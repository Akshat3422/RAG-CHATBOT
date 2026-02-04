from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from utils import clean_text_hard




# Loading the Document
def document_loader(path:str):
    loader=PyPDFLoader(path)
    docs=loader.load()
    return docs



# Splitting the texts into chunks
def splitting_the_text(docs:List[Document]):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    final_chunk_docs=text_splitter.split_documents(documents=docs)
    return final_chunk_docs



# formatting the documents removing extra things other than text
def formatting(final_chunk_docs:List[Document]):
    final_docs=[]
    for doc in final_chunk_docs:
        cleaned_text = clean_text_hard(doc.page_content)
        if cleaned_text.strip():   # empty check
            final_docs.append(cleaned_text)
    return final_docs




