import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.legacy import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

# Load environment variables
load_dotenv()

# Access API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
openai_client = OpenAI(api_key=api_key)

# Directory path containing L1 Docs
directory_path = "Path to L1 Docs"

# Load
documents = SimpleDirectoryReader(directory_path).load_data()

# Embedding model using sentence-transformers
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

# Service context setup
service_context = ServiceContext.from_defaults(chunk_size=1024, embed_model=embed_model)

# VectorStoreIndex for indexing
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

#persist

# Set up query engine
query_engine = index.as_query_engine()

# System Prompt, for GPT-4 Turbo API
system_prompt = """
You are a knowledgeable and accurate Q&A assistant specializing in product support.
Your primary goal is to provide precise and relevant answers and instructions based on the provided knowledge about resolution steps to queries on mobile application and on web application.
If you do not know the answer, it is better to say "I don't know" rather than providing incorrect information.
"""

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# In-memory store for conversation history
conversation_history = []

# Function to interact with GPT-4 Turbo API
def interact_with_gpt(context, query):
    try:
        messages = conversation_history.copy()
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Context: {context}\n\nQuery: {query}"})
        
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            stop=None, #Check
            temperature=0
        )

        gpt_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": gpt_response})
        conversation_history.extend(messages[-2:])  # Update conversation history with last user and assistant messages

        return gpt_response
    
    except Exception as e:
        print(f"Error interacting with GPT-4 Turbo API: {str(e)}")
        return "I'm sorry, I couldn't process your request at the moment."

# Main function to handle queries
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        # Query engine Response
        response = query_engine.query(request.query)
        if response:
            # Retrieve first relevant document
            doc_content = response
            # Generate response with the (GPT-4 Turbo API + context) passed to message
            gpt_response = interact_with_gpt(doc_content, request.query)
            return QueryResponse(response=gpt_response)
        else:
            return QueryResponse(response=interact_with_gpt("", request.query))
    except Exception as e:
        print(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)