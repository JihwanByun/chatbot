# --------------------- FastAPI 설정 부분 --------------------- #

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Any
import nest_asyncio
from pyngrok import ngrok
from threading import Thread
from sembot import Sembot

app = FastAPI()
sembot = Sembot()

model = sembot.get_model()
streamer = sembot.get_streamer()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 특정 도메인으로 제한할 예정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],  # 요청 ID를 노출하도록 설정
)

class SearchRequest(BaseModel):
    question: str

class QueryRequest(BaseModel):
    memory: List[Dict[str, Any]]
    question: str

async def async_stream_response(input_data: dict):
    # stream 출력
    # Thread 객체를 사용하기 때문에 동시성 이슈를 신경써야함
    generation_kwargs = sembot.chain(input_data)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    thread.start()

    for new_text in streamer:
        # print가 아니라 프론트로 전송하는 걸로 바꾸면 될 듯        
        yield new_text.replace("\n", "<br>").replace("<|eot_id|>", "")
        await asyncio.sleep(0.01)  # 스트리밍 중 각 청크 사이의 딜레이 추가

    thread.join()


@app.post("/search")
async def search_documents(query: SearchRequest):
    docs = sembot.docs_invoke(query.question)

    return {"docs": docs}

@app.post("/generate")
async def stream_answer(query: QueryRequest):
    input_data = {
        "memory": query.memory,
        "question": query.question
    }

    return StreamingResponse(async_stream_response(input_data), media_type="text/plain")


# 서버 실행
ngrok_tunnel = ngrok.connect(8001)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8001)