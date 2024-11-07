from typing import Union

from fastapi import FastAPI, UploadFile, File, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# pdf 요약기
from sambot.summarize.summarizer import PDFSummarizer
from sambot.utils.llm_factory import llm_factory
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 특정 도메인으로 제한할 예정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],  # 요청 ID를 노출하도록 설정
)

llm = llm_factory.get_llm("gpt")

summarizer = PDFSummarizer(llm)

@app.post("/summarize")
async def summarize(file: Union[UploadFile, str] = File(...)):
    result = await summarizer.summarize(file, use_parallel=True)
    print(result["summary"])
    return result["summary"]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
