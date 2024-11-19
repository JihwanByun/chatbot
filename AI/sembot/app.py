# app.py

import os
import asyncio
from typing import List, Dict, Any, Union
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from pyngrok import ngrok

import uvicorn
import nest_asyncio

from prepare import load_embedding_model, load_vector_store, load_model_and_tokenizer, filter_docs
from batch import process_queue
from vectorstore_manager import create_vector_store
from summarize.summarizer import PDFSummarizer
from dotenv import load_dotenv

BASE_DIR = ''

EMBEDDINGS_MODEL_PATH = os.path.join(BASE_DIR, "local_embedding")
EMBEDDINGS_MODEL_NAME = 'intfloat/multilingual-e5-large'

VECTORSTORE_PATH = os.path.join(BASE_DIR, "vector_store_path_by_json")

MODEL_PATH = os.path.join(BASE_DIR, "local_model_ncsoft")
MODEL_ID = "NCSOFT/Llama-VARCO-8B-Instruct"

OPENAI_API_KEY='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'

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 특정 도메인으로 제한할 수 있습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# 임베딩 모델, 백터 스토어 로드, 검색기 생성

embeddings = load_embedding_model(EMBEDDINGS_MODEL_PATH, EMBEDDINGS_MODEL_NAME)
vectorstore = load_vector_store(VECTORSTORE_PATH, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
# retriever 객체를 동적으로 관리할 변수 및 Lock 추가
retriever_lock = asyncio.Lock()

# 모델과 토크나이저 로드
tokenizer, model = load_model_and_tokenizer(MODEL_PATH, MODEL_ID)
eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
terminators = [tokenizer.eos_token_id, eos_token_id]

# 요청 큐 및 배치 설정
request_queue = asyncio.Queue()
batch_size = 4  # 필요에 따라 조정 가능
batch_lock = asyncio.Lock()  # 배치 처리 중인지 확인하기 위한 Lock

# 이벤트 루프를 저장할 변수
event_loop = None

class QueryRequest(BaseModel):
    memory: List[Dict[str, Any]]
    question: str
    level: int

class SearchRequest(BaseModel):
    question: str
    level: int

class UpdateRequest(BaseModel):
    regulationList: List[Dict[str, Any]]
    text: str
    level: int
    title: str

# 앱 시작 시 이벤트 루프 저장 및 배치 프로세서 실행
@app.on_event("startup")
async def startup_event():
    global event_loop
    event_loop = asyncio.get_running_loop()
    asyncio.create_task(process_queue(request_queue, batch_size, batch_lock, retriever_lock, event_loop, tokenizer, model, terminators, embeddings, retriever), name="process_queue")


@app.post("/search")
async def search_documents(query: SearchRequest):    

    global retriever  # 전역 retriever 접근
    
    docs = retriever.invoke(query.question)

    print("question: ", query.question)

    print()

    print("search: ", docs)

    filtered_docs = filter_docs(docs, query.level)

    return {"docs": filtered_docs}


@app.post("/generate")
async def generate_text(query: QueryRequest):
    # 각 요청에 대한 응답 큐 생성
    response_queue = asyncio.Queue()
    # 요청 큐에 추가
    await request_queue.put({'question': query.question, 'memory': query.memory, 'level': query.level, 'response_queue': response_queue})

    # 스트리밍 제너레이터 함수
    async def response_generator():
        while True:
            token = await response_queue.get()
            if token is None:
                break
            yield token.replace("\n", "<br>")  # 클라이언트에게 부분 응답을 스트리밍으로 전송

    # StreamingResponse를 이용해 스트리밍 응답 반환
    return StreamingResponse(response_generator(), media_type="text/plain")


@app.post("/update")
async def update_regulations(request: UpdateRequest):       
    pdf_file = {
        "title": request.title,
        "content": request.text,
        "level": request.level
    }

    global retriever  # 전역 retriever 접근

    vectorstore = create_vector_store(VECTORSTORE_PATH, embeddings, pdf_file, request.regulationList)

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})

    for task in asyncio.all_tasks():
        if task.get_name() == "process_queue":
            task.cancel()

    asyncio.create_task(process_queue(request_queue, batch_size, batch_lock, retriever_lock, event_loop, tokenizer, model, terminators, embeddings, retriever), name="process_queue")

    print("반영했습니다.")

    return True


@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

    summarizer = PDFSummarizer(llm)
    
    result = await summarizer.summarize(file, use_parallel=True)
    
    return result["summary"]
    
# 서버 실행
ngrok.set_auth_token("2p11sAfF3sDVEc1e8D1IluklIyE_4zo5qyuQ6RifHzMAcf4K9")
ngrok_tunnel = ngrok.connect(8000, domain="flowing-mongoose-exotic.ngrok-free.app")
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)