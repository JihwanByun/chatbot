import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 설정 부분
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vector_store_path")

# Load the embedding model
print("임베딩 모델 가져오는중")
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-nli")

# Load the FAISS vectorstore
print("백터 스토어 가져오는중")
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

# 검색어에 대해 검색 수행 및 스코어와 함께 문서 반환
# query = "안녕하세요"
# query = "동의하십니까"
# query = "개인정보 관련 수집은 어떻게 하나요?"
# query = "알려 드립니다"
# query = "연차 신청은 어떻게 하나요?"
query = "청원심의회서면의결서"

docs = retriever.invoke(query)

print(docs)