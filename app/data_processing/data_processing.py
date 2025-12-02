from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from typing import List
from langchain_core.documents import Document

class ProcessData:
    """Process a saved file on disk and return a list of text chunks (langchain Documents)."""

    SUPPORTED_EXT = (".pdf")

    def __init__(self, file_path: str):
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string path to an existing file")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")
        self.file_path = file_path
        self.filename = os.path.basename(file_path)

    def _is_supported(self) -> bool:
        return self.filename.lower().endswith(self.SUPPORTED_EXT)

    def process_document(self) -> List[Document]:
        if not self._is_supported():
            raise ValueError(f"Unsupported file type for {self.filename}")

       
        if self.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()

        else:
    
            raise ValueError("Unsupported file type")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        return chunks