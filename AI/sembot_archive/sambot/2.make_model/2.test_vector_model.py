# --------------------- todo --------------------- #

# 1. 1번 파일의 이슈 해결
# 2. langchain 메모리가 아닌 메모리 변수로 사용했을 때 로컬 모델 아에 에러남.
#    메모리 변수로 바꾸고 에러 해결

# --------------------- 설정 부분 --------------------- #

import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vector_store_path")
MODEL_PATH = os.path.join(BASE_DIR, "local_model")
MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"


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

<previous_memory>
{memory}

<search_results>
{search_results}

<user>
{question}

<assistant>
"""
)


# # --------------------- local LLM 다운로드 --------------------- #

# # LLM모델 없으면 다운받기
# from huggingface_hub import snapshot_download

# if not os.path.exists(MODEL_PATH) or len(os.listdir(MODEL_PATH)) == 0:
#     print("LLM모델이 현재 경로에 없어 다운중입니다...")
#     snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_PATH, local_dir_use_symlinks=False)


# # --------------------- local LLM 만드는 부분 --------------------- #

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_huggingface.llms import HuggingFacePipeline

# # 토크나이저와 모델 불러오기
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True, 
#     torch_dtype=torch.bfloat16,  # bfloat16을 사용하여 메모리 절약
#     device_map="auto",
#     # max_memory={0: "16GB"}
# )

# # 파이프라인 구성
# pipe = pipeline(
#     "text-generation", 
#     model=model, 
#     tokenizer=tokenizer,
#     # max_new_tokens=20,  # 생성할 토큰 수 설정
#     temperature=0.6,  # 텍스트 생성에 사용되는 온도값
#     top_p=0.9,  # top-p 샘플링
#     return_full_text=False,  # 답변만 출력
#     eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"), "<end>", "<assistant>"]  # 문장 종료 토큰 설정
# )

# # HuggingFacePipeline을 사용하여 LLM 인스턴스 생성
# llm = HuggingFacePipeline(pipeline=pipe)


# --------------------- GPT LLM --------------------- #

# GPT LLM 추가 (모델 확장성 고려 및 컴퓨팅 파워 한계)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --------------------- chain 만드는 부분 --------------------- #

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"참고문서:\n{context}\n") # 추후 소켓방식 변경예정
    return context

def create_conversarion_memory(memory_key, output_key):
    memory = ConversationBufferWindowMemory(
        return_messages=True,
        memory_key=memory_key,
        output_key=output_key
    )
    return memory

memory = create_conversarion_memory("chat_history", "answer")

chain = (
    {
        "memory": lambda x: memory.load_memory_variables({})["chat_history"],
        "search_results": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --------------------- chain 테스트 부분 --------------------- #

# 스트리밍 답변을 위한 함수
def stream_response(input_data):
    answer=''
    for chunk in chain.stream(input_data):
       print(chunk, end="", flush=True)
       answer += chunk
    print("\n\n")
    return answer

# 질문 예시
question = "유류비 청구는 어떻게 할 수 있나요? 제가 3박 4일로 부산에 갓다오는데 비용은 대충 어떻게 계산하죠?"
answer = stream_response(question)
memory.save_context({"question": question}, {"answer": answer})

question = "연차는 어떻게 신청할 수 있나요?"
answer = stream_response(question)
memory.save_context({"question": question}, {"answer": answer})

question = "아까 제가 어떤 질문 했죠?"
answer = stream_response(question)
memory.save_context({"question": question}, {"answer": answer})