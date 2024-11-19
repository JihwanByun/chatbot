# --------------------- todo --------------------- #

# make_model 에 있는 파일들 이슈 해결해서 로컬모델로 동작 되게 해야함.


# --------------------- 설정 부분 --------------------- #

import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vector_store_path")
MODEL_PATH = os.path.join(BASE_DIR, "local_model")

# --------------------- 검색기 만드는 부분 --------------------- #

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 임베딩 모델 가져오기
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-nli")

# 백터스토어 가져오기
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
)

# 검색기 생성
retriever = vectorstore.as_retriever()

# --------------------- 프롬프트 만드는 부분 --------------------- #

from langchain_core.prompts import PromptTemplate

# 프롬프트 템플릿 작성
prompt = PromptTemplate.from_template(
    """<system>
"You are a helpful AI assistant. Please answer the user's questions simply and kindly. 당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 간단하고 친절하게 답변해주세요."

<conversation_history>
{memory}
</conversation_history>

<previous_search_results>
{memory_retriever}
</previous_search_results>

<current_search_results>
{retriever}
</current_search_results>

<user>
{question}
</user>

<assistant>
"""
)

# --------------------- GPT LLM --------------------- #

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------- chain 만드는 부분 --------------------- #

from langchain.schema import Document  # Document 객체 사용
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda


def format_memory(memory):
    conversation = ""
    for i, qda in enumerate(memory, start=1):
        # qda에서 question 추출
        question = qda.get('question', '')
        if question:
            conversation += f"User Before{i + 1}: {question}\n"
        # qda에서 answer 추출
        answer = qda.get('answer', '')
        if answer:
            conversation += f"Assistant Before{i + 1}: {answer}\n"
        conversation += "\n"
    return conversation.strip()


def format_docs(docs):
    context = ""
    for i, doc in enumerate(docs, start=1):
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown source')
        context += f"Document {i} (Source: {source}):\n{content}\n"
    return context.strip()


def memory_retriever_func(inputs):
    memory = inputs["memory"]
    question = inputs["question"]
    docs = []
    for qda in memory:
        # json 인자 추출 부분
        question_text = qda.get('question', '')
        doc_content = qda.get('doc', {}).get('page_content', '')
        answer_text = qda.get('answer', '')

        content = f"Question: {question_text}\n"
        if doc_content:
            content += f"Document Content: {doc_content}\n"
        content += f"Answer: {answer_text}"
        docs.append(Document(page_content=content))

    if not docs:  # docs 없으면 없는거로 반환
        return []

    memory_vectorstore = FAISS.from_documents(docs, embeddings)  # 벡터 스토어 만들기
    memory_retriever = memory_vectorstore.as_retriever()  # 메모리 검색기 생성
    retrieved_docs = memory_retriever.invoke(question)  # 메모리 검색

    return retrieved_docs  # 검색 결과 반환


chain = (
        RunnableMap(
            {
                "memory": RunnableLambda(lambda x: x["memory"]) | format_memory,
                "memory_retriever": RunnableLambda(memory_retriever_func) | format_docs,
                "retriever": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
                "question": RunnableLambda(lambda x: x["question"]),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
)

# --------------------- FastAPI 설정 부분 --------------------- #

from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import asyncio

# pdf 요약기
from summarize.summarizer import PDFSummarizer

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


class SearchRequest(BaseModel):
    question: str


class QueryRequest(BaseModel):
    memory: List[Dict[str, Any]]
    question: str


async def async_stream_response(input_data: dict):
    # invoke 출력
    # answer = chain.invoke(input_data).replace("\n", "<br>") # 엔터 두번 방지
    # yield answer

    # stream 출력
    for chunk in chain.stream(input_data):
        yield chunk.replace("\n", "<br>")  # 엔터 두번 방지
        await asyncio.sleep(0.01)  # 스트리밍 중 각 청크 사이의 딜레이 추가


@app.post("/search")
async def search_documents(query: SearchRequest):
    docs = retriever.invoke(query.question)

    return {"docs": docs}


@app.post("/generate")
async def stream_answer(query: QueryRequest):
    input_data = {
        "memory": query.memory,
        "question": query.question
    }

    return StreamingResponse(async_stream_response(input_data), media_type="text/plain")


summarizer = PDFSummarizer(llm)

@app.post("/summarize")
async def summarize(file: Union[UploadFile, str] = File(...)):
    result = await summarizer.summarize(file, use_parallel=True)
    print(result["summary"])
    return result["summary"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
