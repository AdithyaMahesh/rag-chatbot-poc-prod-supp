from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage, Document
from llama_index.legacy.llms.openai import OpenAI
import os
from dotenv import load_dotenv
import json
import shutil

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# Directory paths containing L1 and L2 Docs
l1_directory_path = "C:\\Personal\\Internship\\Sumeru\\RAG\\Demo\\NewData\\Data\\L1" # Replace with L1 folder location
l2_directory_path = "C:\\Personal\\Internship\\Sumeru\\RAG\\Demo\\NewData\\Data\\L2" # Replace with L2 folder location
persist_dir = "./index"

system_prompt = """
You are a knowledgeable and accurate Q&A assistant specializing in product support.
Your goal is to provide precise and relevant answers and instructions based on the provided knowledge about resolution steps to queries on mobile applications and web applications for L1 , and SQL queries for L2/Database queries.
If you do not know the answer or the question isn't relevant to knowledge base, it is better to say "I don't know" rather than providing incorrect information.
"""

def load_data_from_directory(base_path, subdirs):
    documents = []
    for subdir in subdirs:
        full_path = os.path.join(base_path, subdir)
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        metadata = {
                            'source': subdir,
                            'file_name': file
                        }
                        document = Document(text=content, metadata=metadata)
                        documents.append(document)
    return documents

def load_all_documents():
    l1_subdirs = ['web', 'mobile']
    l2_subdirs = ['queries']

    l1_documents = load_data_from_directory(l1_directory_path, l1_subdirs)
    l2_documents = load_data_from_directory(l2_directory_path, l2_subdirs)

    return l1_documents + l2_documents

def create_index():
    documents = load_all_documents()

    model = "gpt-3.5-turbo"
    llm = OpenAI(api_key=api_key, model=model, temperature=0, system_prompt=system_prompt, streaming=True)

    # Create an index
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=persist_dir)

    print(f"Using model: {model}")
    return index

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def create_or_load_index():
    if not os.path.exists(persist_dir):
        print("Creating New Index")
        return create_index()
    else:
        print("Loading the Persisted Index")
        return load_index()

index = create_or_load_index()

# Make sure chat_engine is updated globally after the index is rebuilt
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=False)

app = FastAPI()

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

@app.post("/update_index")
async def update_index():
    global index, chat_engine
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # Remove the old index directory
    index = create_index()
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=False)  # Update chat_engine
    return {"message": "Index updated successfully"}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
