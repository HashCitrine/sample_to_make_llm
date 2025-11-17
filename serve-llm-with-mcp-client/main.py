from fastapi import FastAPI, Request
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from vllm import VLLM
from pydantic import BaseModel

import os

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

model_path = "/path/to/model.gguf"
model = VLLM(model_path=model_path, enable_auto_tool_choice=True)

# mcp_servers = ["http://mcp-server1:8080", "http://mcp-server2:8080"]
mcp_servers = os.getenv("MCP_SERVERS", "").split(sep=",")

client = None
agent = None

@app.on_event("startup")
async def startup_event():
    global client, agent
    client = await MultiServerMCPClient(server_urls=mcp_servers).__aenter__()
    agent = create_react_agent(model, client.get_tools())

@app.on_event("shutdown")
async def shutdown_event():
    global client
    await client.__aexit__(None, None, None)

@app.post("/api/query")
async def query(request: QueryRequest):
    response = await agent.ainvoke({"messages": request.question})
    final_answer = [msg for msg in response["messages"] if msg.__class__.__name__ == "AIMessage"][-1].content
    return {"answer": final_answer}


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True
    )