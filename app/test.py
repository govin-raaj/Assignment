
from app.vector_store.vector_store import VectorStore

from langchain_groq import ChatGroq
from app.config import Config
from app.data_processing.doc_processing import Processdoc



vector_store = VectorStore()

file_path="D:/ML/Assignment/data/qatar_test_doc.pdf"

processdoc=Processdoc(file_path,vector_store)

docs=processdoc.process_documents()

llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=Config.qroq_api_key,
        )

query="What are Qatar's GDP growth drivers for 2024-2025?"

context_docs = processdoc.retrieve_multimodal(query, k=5)

message = processdoc.create_multimodal_message(query, context_docs)

response = llm.invoke([message])

print("context_doc-----------------------------------")
print(context_docs)

print("message-----------------------------------------")
print(message)
print("llm mssg--------------------------------------")
print(response.content)



