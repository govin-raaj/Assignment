
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np


class VectorStore:
    def __init__(self, path="faiss_store"):
        
        self.path = path
        os.makedirs(path, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        index_path = os.path.join(path, "faiss.index")

        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(
                folder_path=path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None



    def add_documents(self, documents):
       
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_documents(documents)

        self.vector_store.save_local(self.path)



    def similarity_search(self, query, k=4):
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
    


    def as_retriever(self, search_kwargs=None):
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Add documents first.")
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    

    
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

        if hasattr(self.vector_store, "similarity_search_by_vector"):
            return self.vector_store.similarity_search_by_vector(emb, k=k)

        return self.vector_store.similarity_search_by_vector(emb, k=k)