from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.legacy.llms.openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# Directory path containing L1 Docs
directory_path = "C:\\Personal\\Internship\\Sumeru\\RAG\\Adithya\\Adithya\\L1_Docs"

# Initialize OpenAI client with streaming enabled
openai_client = OpenAI(api_key=api_key, streaming=True)

# Load documents
documents = SimpleDirectoryReader(directory_path).load_data()

# System Prompt, for GPT-4 Turbo API
system_prompt = """
You are a knowledgeable and accurate Q&A assistant specializing in product support.
Your primary goal is to provide precise and relevant answers and instructions based on the provided knowledge about resolution steps to queries on mobile application and on web application.
If you do not know the answer, it is better to say "I don't know" rather than providing incorrect information.
"""

model = "gpt-4"


llm = OpenAI(model=model, temperature=0, system_prompt=system_prompt, streaming=True)

# Create an index
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=False)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware to allow frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_response(prompt):
    response = chat_engine.stream_chat(prompt)
    for token in response.response_gen:
        yield f"data: {json.dumps({'response': token})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    return StreamingResponse(generate_response(prompt), media_type="text/event-stream")
