from app.vector_store.vector_store import VectorStore
from langchain_groq import ChatGroq
from app.config import Config
from app.data_processing.doc_processing import Processdoc
import streamlit as st
import os


llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=Config.qroq_api_key,
        )

file_path="D:/ML/Assignment/data/qatar_test_doc.pdf"

def load_pdf_pipeline():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at {file_path}")

    vs = VectorStore()
    proc = Processdoc(file_path,vs)
    docs=proc.process_documents()
    return proc, vs, docs

if "proc" not in st.session_state:
    with st.spinner("Processing PDF and building multimodal index (runs once)..."):
        proc, vector_store, docs = load_pdf_pipeline()
        st.session_state["proc"] = proc
        st.session_state["vector_store"] = vector_store
        st.session_state["docs"] = docs
else:
    proc = st.session_state["proc"]
    vector_store = st.session_state["vector_store"]
    docs = st.session_state["docs"]



# with st.spinner("Processing PDF and building multimodal index..."):
#     proc, vector_store, docs = load_pdf_pipeline()
# st.success(f"Loaded {len(docs)} chunks/images from PDF")


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.spinner("Retrieving context..."):
        context_docs = proc.retrieve_multimodal(user_input, k=5)
        message = proc.create_multimodal_message(user_input, context_docs)

        response = llm.invoke([message])
    
    ai_message = response.content

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)