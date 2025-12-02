from langchain_core.messages import BaseMessage, SystemMessage,HumanMessage
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from app.data_processing.data_processing import ProcessData
from app.vector_store.vector_store import VectorStore
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from typing import TypedDict, Annotated,List 
from langgraph.graph import StateGraph, START
from fastapi.params import Form as form
from langchain_groq import ChatGroq
from app.config import Config
import os
import uvicorn


app = FastAPI()

vector_store = VectorStore()

file_path="D:/ML/Assignment/data/qatar_test_doc.pdf"
processor = ProcessData(file_path)
chunks = processor.process_document()
vector_store.add_documents(chunks)


llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=Config.qroq_api_key,
        )

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



@tool
def retrieval(query: str) -> List[dict]:
    """
    Retrieve relevant information from the uploaded PDF(s).
    """
    if vector_store.vector_store is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    try:
        retr = vector_store.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retr.invoke(query)  
    except Exception:

        docs = vector_store.similarity_search(query, k=3)

    context = [getattr(d, "page_content", str(d)) for d in docs]
    metadata = [getattr(d, "metadata", {}) for d in docs]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
    }


tools=[retrieval]

llm_with_tool=llm.bind_tools(tools)

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    system_message = SystemMessage(
        content=(
        "You are a helpful assistant. When the user asks about the uploaded document, "
        "use the retriever tool to fetch relevant document passages. "
        "The retrieval tool returns a list of objects with 'page_content' and 'metadata'. "
        "If there are no documents available, ask the user to upload the PDF. "
        "If you genuinely do not know the answer, say you don't know."
        )
    )

    messages = [system_message,*state['messages']]
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)



@app.get("/", response_class=HTMLResponse) 
async def home(request: Request): 
    return JSONResponse({"home page url"})

@app.post("/query")
async def query_rag(query:str= form(...)):
    try:
        initial_messages = [HumanMessage(content=query)]
        initial_state = {"messages": initial_messages}
        CONFIG = {"configurable": {"thread_id": "thread_1"}}
        response= chatbot.invoke(initial_state,config=CONFIG)
        ai_message = response['messages'][-1].content
        return JSONResponse({"response": ai_message}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
    

@app.get("/vectorstore_result")
def vectorstore_result(query: str):
    try:
        results = vector_store.similarity_search(query, k=4)
        formatted_results = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in results
        ]
        return JSONResponse({"results": formatted_results}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during similarity search: {e}")
    

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)