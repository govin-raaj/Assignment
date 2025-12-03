
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np


class VectorStore:
    def __init__(self, path="faiss_store"):
        
        self.path = path
        os.makedirs(path, exist_ok=True)

        index_path = os.path.join(path, "faiss.index")

        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(
                folder_path=path,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None
    
    def add_embeddings(self, documents, embeddings):
        """
        Add documents + precomputed embeddings (numpy arrays).
        embeddings: list/array of shape (N, D)
        """
        embeddings = np.array(embeddings).astype("float32")

        if self.vector_store is None:
    
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=[
                    (doc.page_content, emb) for doc, emb in zip(documents, embeddings)
                ],
                embedding=None,
                metadatas=[doc.metadata for doc in documents]
            )
        else:
           
            self.vector_store.add_embeddings(
                text_embeddings=[
                    (doc.page_content, emb) for doc, emb in zip(documents, embeddings)
                ],
                embedding=None,
                metadatas=[doc.metadata for doc in documents]
            )

        self.vector_store.save_local(self.path)


    def similarity_search_by_vector(self, embedding, k=4):
        """
        Accepts a numpy vector (1D) or list and performs similarity search.
        Delegates to underlying FAISS wrapper if available.
        """
        if self.vector_store is None:
            return []

        emb = np.array(embedding).astype("float32")
        
        return self.vector_store.similarity_search_by_vector(emb, k=k)