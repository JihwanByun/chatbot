import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Document 객체 사용
from langchain_huggingface import HuggingFaceEmbeddings

# 설정 부분
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
JSON_PATH = os.path.join(BASE_DIR, '1.make_vector_store\kisa.json') # 변환된 JSON 파일 경로
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vector_store_path")  # 생성된 벡터스토어를 저장할 경로
EMBEDDINGS_MODEL_NAME = 'jhgan/ko-sroberta-nli'  # 임베딩에 사용할 모델
EMBEDDINGS_MODEL_KWARGS = {'device': 'cuda'}  # 임베딩 모델 로드 시 추가 설정 (GPU 사용)
EMBEDDINGS_ENCODE_KWARGS = {'normalize_embeddings': True}  # 임베딩 벡터 생성 시 설정 (정규화)

# JSON 데이터를 벡터스토어로 변환하는 함수
def create_vectorstore_from_json(json_path, vectorstore_path, embeddings):
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print("백터 DB를 만드는 중...")
    # JSON 데이터를 Document 형식으로 변환
    all_documents = []
    for doc in documents:
        document = Document(
            metadata=doc["metadata"],
            page_content=doc["page_content"]
        )
        all_documents.append(document)

    # 분할된 문서들을 임베딩하여 벡터스토어 생성
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    vectorstore.save_local(vectorstore_path)  # 벡터스토어를 로컬에 저장

    return vectorstore

def main():
    print("임베딩 모델을 가져옵니다...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=EMBEDDINGS_MODEL_KWARGS,
        encode_kwargs=EMBEDDINGS_ENCODE_KWARGS
    )
    
    vectorstore = create_vectorstore_from_json(JSON_PATH, VECTORSTORE_PATH, embeddings)
    print("벡터 스토어가 로컬에 생성 및 저장되었습니다.")

if __name__ == "__main__":
    main()
