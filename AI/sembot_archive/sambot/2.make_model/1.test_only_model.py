# --------------------- todo --------------------- #

# 1. local LLM 스트리밍 멈추지 않음 이슈 해결

# --------------------- 설정 부분 --------------------- #

import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
MODEL_PATH = os.path.join(BASE_DIR, "local_model")
MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"


# --------------------- 프롬프트 만드는 부분 --------------------- #

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """<system>
"You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 친절하게 답변해주세요."

<user>
{question}
</user>

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
#     device_map="auto"  # 모델을 GPU에 분배
#     # max_memory={0: "16GB"}
# )

# # 파이프라인 구성
# pipe = pipeline(
#     "text-generation", 
#     model=model, 
#     tokenizer=tokenizer, 
#     max_new_tokens=20,  # 생성할 토큰 수 설정
#     temperature=0.6,  # 텍스트 생성에 사용되는 온도값
#     top_p=0.9,  # top-p 샘플링
#     return_full_text=False,  # 답변만 출력
#     eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # 문장 종료 토큰 설정
# )

# # HuggingFacePipeline을 사용하여 LLM 인스턴스 생성
# llm = HuggingFacePipeline(pipeline=pipe)


# --------------------- GPT LLM --------------------- #

# # GPT LLM 추가 (모델 확장성 고려 및 컴퓨팅 파워 한계)
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --------------------- chain 만드는 부분 --------------------- #

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --------------------- chain 테스트 부분 --------------------- #

question = "서울의 유명한 관광 코스를 만들어줄래?"

# invoke 테스트
# print(chain.invoke({"question": question}))

# stream 테스트
def stream_response(input_data):
    answer = ''
    for chunk in chain.stream(input_data):
        print(chunk, end="", flush=True)
        answer += chunk
    return answer

answer = stream_response(question)