
import os
os.environ.setdefault(
    "USER_AGENT",
    "AdvancedRAG/1.0 (Prajwal Patil)"
)


from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.chains import  create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq




from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)


loader = PyPDFLoader('company_policy.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()
texts = [c.page_content for c in chunks]

embed_docs = embeddings.embed_documents(texts)


from pinecone import Pinecone, ServerlessSpec


API_KEY = os.getenv("PINECONE_API_KEY")


pc = Pinecone(api_key = API_KEY)

index_name = "rag-index-3"

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name = index_name,
        dimension = 768,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        )
    )

index = pc.Index(index_name)

import uuid

data = []

for i,vector in enumerate(embed_docs):
    data.append((
        str(uuid.uuid4()),
        vector,
        {"text": chunks[i].page_content}
    ))

index.upsert(vectors=data, namespace="resume_v1")

## QUERY REWRITING 
def query_rewriting(query):

    prompt = f"""
You are a query rewriter for a retrieval system.

Your task:
Rewrite the query to be MORE CONCISE and MORE PRECISE.

STRICT RULES:
1. Do NOT change the original intent.
2. Do NOT add new information or assumptions.
3. Do NOT broaden the scope.
4. Do NOT make the query longer unless required for clarity.
5. If the query is already concise and clear, return it EXACTLY as-is.

Allowed actions:
- Remove filler words.
- Resolve obvious pronouns (only if clearly implied).
- Expand acronyms ONLY if obvious.

Output rules:
- Return ONLY the rewritten query.
- No explanations.
- No quotes.

User query:
{query}
"""
     
    response = model.invoke(prompt)
    return response.content



## Hybrid Reterival -> Semantic Search in Pinecone and Keywords Search through BM25 in chunks.
## Created BM25Retriever on Chunks.
from langchain_community.retrievers import BM25Retriever
bm25 = BM25Retriever.from_documents(chunks)

def hybrid_reterival(query):
    query_vector = embeddings.embed_query(query)

    results_semantic = index.query(
        vector=query_vector,
        top_k=20,
        include_metadata=True
    )

    # vector_data = [r.matches[i]['metadata']['text'] for r in results]

    vector_docs = [] # data got from semantic search
    for r in results_semantic.matches:
        vector_docs.append(r['metadata']['text'])



    results_keyword = bm25.invoke(query)

    keyword_docs =[]

    for r in results_keyword:
        keyword_docs.append(r.page_content)

    candidates = list(set(vector_docs + keyword_docs))
    return candidates


## RE-RANKING 
# We will use cross-encoder for re-ranking the candidates
# Initialzing reranker
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def re_ranking(candidates,new_query):
    
    pairs = []
    for c in candidates:
        pairs.append([new_query,c])
    scores = reranker.predict(pairs)
    

    # pair scores with documents, sort by score descending, and extract docs
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    ranked_docs = [doc for _, doc in scored_docs]
    top_docs = ranked_docs[:4]

    if not top_docs:
       return "I don't have enough information to answer this."

    return top_docs


## Context Assembly
def context_assembly(top_docs):
    context = "\n\n".join([doc for doc in top_docs])
    return context



# Saving the Logs and generating the response from model
import time
import json
import uuid
from datetime import datetime

model_cost = {
    "openai" : 0.00003
}


# creating a pydantic model to validate and take response from frontend.
from pydantic import BaseModel
from fastapi.responses import StreamingResponse


class ChatRequest(BaseModel):
    query: str

# creating endpoint for chatting to the backend
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] for strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"message" : "App is running fine"}

@app.get("/home")
def root():
    return {"message" : "App successfully Loaded"}


class LLMGaurd:
    def log(self, record:dict):
        with open('rag_logs.json','a') as f:
            f.write(json.dumps(record) + "\n")

    def log_eval(self,record: dict):
        with open('eval_logs.json','a') as f:
            f.write(json.dumps(record) + "\n")


# Used for streaming and logging and retry and backoff logic
import time
import uuid
from datetime import datetime
from evaluation import context_relevator_eval,context_sufficient_eval,faithfulness_eval
from evaluation import answer_relevance_evaluation, answer_completeness_eval


import tiktoken

def count_tokens(text: str, model="cl100k_base"):
    """
    Count tokens in a text using a given tiktoken encoding.
    For custom / OSS models, use an explicit encoding name like 'cl100k_base'.
    """
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))



def llm_response(
    context: str,
    query: str,
    logger: LLMGaurd,
    max_retries: int = 3,
    backoff_base: int = 2
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    final_prompt = f"""
    Use the context below to answer accurately.

    Context:
    {context}

    Question:
    {query}
    """

    attempt = 0

    while attempt < max_retries:
        try:

            response = model.invoke(final_prompt)
            print(response.content)

            end_time = time.time()

            usage = response.response_metadata.get("token_usage", {})
            if usage:
                total_tokens = usage.get("total_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                
            else:
                prompt_tokens = count_tokens(final_prompt)
                completion_tokens = count_tokens(response.content)
                total_tokens = prompt_tokens + completion_tokens

            cost = round(total_tokens * 0.00003,3) # this is total cost per question and response both.

            logger.log({
                    "request_id": request_id,
                    "model": "openai",
                    "question": query,
                    "latency": round(end_time - start_time, 3),
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost,
                    "status": "success",
                    "timestamp": str(datetime.utcnow())
                })
            break
            
        except Exception as e:
            attempt += 1

            if attempt >= max_retries:
                # ❌ Final failure
                logger.log({
                    "request_id": request_id,
                    "model": "openai",
                    "status": "failed",
                    "error": str(e),
                    "attempts": attempt,
                    "timestamp": str(datetime.utcnow())
                })
                raise

            # 🔁 Retry with backoff
            time.sleep(backoff_base ** attempt)
            

    return response.content,request_id


def evaluate_response(request_id,query,context,response,logger:LLMGaurd):

    eval1 = context_relevator_eval(query, context)
    eval2 = context_sufficient_eval(query, context)
    eval3 = faithfulness_eval(response, context)
    eval4 = answer_relevance_evaluation(query, response)
    eval5 = answer_completeness_eval(query, response)

    logger.log_eval({
                    "request_id": request_id,
                    "context_relevance": {
                        "relevance_score": eval1.context_relevance_score,
                        "explanation": eval1.explanation
                    },
                    "is_context_sufficient": {
                        "sufficient": eval2.sufficiency,
                        "missing_information": eval2.missing_information
                    },
                    "is_response_faithfull": {
                        "faithful": eval3.faithful,
                        "claims": eval3.unsupported_claims
                    },
                    "is_answer_relevant": {
                        "relevance_score": eval4.answer_relevance_score,
                        "explanation": eval4.explanation
                    },
                    "is_answer_complete": {
                        "completeness": eval5.completeness,
                        "missing_parts": eval5.missing_parts
                    }
    })

    return


from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse

    
@app.post("/chat")
def chat(request: ChatRequest,background_tasks: BackgroundTasks):
    query = request.query
    new_query = query_rewriting(query)
    candidates = hybrid_reterival(new_query)
    top_docs = re_ranking(candidates, new_query)
    context = context_assembly(top_docs)

    logger = LLMGaurd()


    response, request_id = llm_response(context, new_query, logger)

    background_tasks.add_task(evaluate_response,request_id,query,context,response,logger)

    return JSONResponse({"response":response})


@app.get("/logs/rag")
def get_rag_logs():
    try:
        with open("rag_logs.json") as f:
            logs = [json.loads(line) for line in f]
        return JSONResponse(logs)
    except Exception:
        return JSONResponse([])

@app.get("/logs/eval")
def get_eval_logs():
    try:
        with open("eval_logs.json") as f:
            logs = [json.loads(line) for line in f]
        return JSONResponse(logs)
    except Exception:
        return JSONResponse([])

        


