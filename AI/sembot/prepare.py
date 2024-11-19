# prepare.py

import os
import json
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(path, name):
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        print("LLM모델이 현재 경로에 없어 다운중입니다...")
        os.makedirs(path)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="balanced")
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="balanced")
    model.eval()

    # pad_token_id 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def load_embedding_model(path, name):
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        print("임베딩모델이 현재 경로에 없어 다운중입니다...")
        snapshot_download(repo_id=name, local_dir=path, local_dir_use_symlinks=False)
    embeddings = HuggingFaceEmbeddings(model_name=path)

    return embeddings


def load_vector_store(path, embeddings):
    vectorstore = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )

    return vectorstore


def load_prompt():
    prompt = """You are a knowledgeable AI assistant trained to answer questions about company regulations. Respond directly and concisely, using the information from company policies. Answer as follows: '사내 규정에 따르면...' and provide a clear response based on the retrieved content.

    당신은 회사 규정에 대해 질문을 명확하고 직접적으로 답변하는 유능한 AI 어시스턴트입니다. 답변은 '사내 규정에 따르면...'의 형식으로 시작하고 수학연산은 하지 않으며 검색된 내용에 따라 명확한 답을 제공하세요."""

    return prompt


def format_docs(docs):
    context = ""
    for i, doc in enumerate(docs, start=1):
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown source")
        context += f"Document (Source: {source}):\n{content}\n\n"
    context = context.strip()

    return context


def memory_retriever(input_data, embeddings):
    memory = input_data["memory"]
    question = input_data["question"]
    docs = []

    # 메모리 데이터를 JSON 형식으로 바꾸기
    for qda in memory:
        doc = qda.get("doc", {})
        # 검색데이터가 없으면 저장 안하도록
        if not doc:
            continue

        qda_json = {
            "question": qda.get("question", ""),
            "doc": doc,
            "answer": qda.get("answer", ""),
        }
        docs.append(Document(page_content=json.dumps(qda_json)))

    if not docs:  # docs 없으면 없는거로 반환
        return []

    # FAISS 벡터 스토어 생성 및 검색기 구성
    memory_vectorstore = FAISS.from_documents(docs, embeddings)
    memory_retriever = memory_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.75})

    # 질문과 관련 있는 qda 유사도 검색
    retrieved_docs = memory_retriever.invoke(question)

    # 검색된 qda 에서 d만 추출하고 따로 저장
    extracted_docs = []
    for doc in retrieved_docs:
        # JSON 파싱 후 doc 내용 추출
        document_data = json.loads(doc.page_content)
        doc_content = document_data.get("doc", {}).get("page_content", "")
        doc_metadata = document_data.get("doc", {}).get("metadata", {})

        # doc 내용과 메타데이터로 새로운 Document 객체 생성
        extracted_docs.append(
            Document(page_content=doc_content, metadata=doc_metadata)
        )

    return extracted_docs  # Document 객체 리스트 반환

def filter_docs(docs: list, level):

      filtered_docs = []

      for index, doc in enumerate(docs):
          if int(doc.metadata["level"]) > int(level):
              if index == 0:
                  return False
              else:
                  continue

          filtered_docs.append(doc)

      return filtered_docs