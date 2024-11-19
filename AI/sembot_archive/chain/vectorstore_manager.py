import os

from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS, MongoDBAtlasVectorSearch
from config import MONGODB_CLUSTER_URI, MONGODB_DB_NAME
from mongo_client import MongoDBClient


# PDF 파일명들을 로드하는 함수
def load_pdf_files(pdf_path):
    """주어진 디렉토리에서 모든 PDF 파일들을 로드합니다."""
    pdf_files = []
    for root, dirs, files in os.walk(pdf_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


# PDF를 JSON으로 변환하는 함수
def pdf_splitter(pdf_path):
    pdf_files = load_pdf_files(pdf_path)
    all_documents = []

    for pdf_file in pdf_files:
        loader = PDFPlumberLoader(pdf_file)
        documents = loader.load()  # PDF 문서를 로드합니다

        # 텍스트를 작은 청크로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 각 청크의 최대 문자 수
            chunk_overlap=32,  # 청크 간 중복되는 문자 수
        )
        chunked_documents = text_splitter.split_documents(documents)
        all_documents.extend(chunked_documents)

    return all_documents


def get_connection():
    # MongoDBClient 인스턴스를 애플리케이션에서 전역적으로 사용
    mongo_client = MongoDBClient(uri=MONGODB_CLUSTER_URI, db_name=MONGODB_DB_NAME)
    collection = mongo_client.get_collection()

    return collection


def create_vectorstore_from_Mongo(pdf_path, embeddings):

    # create mongodb vectore store
    collection = get_connection()
    chunked_docs = pdf_splitter(pdf_path)

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=chunked_docs,
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        relevance_score_fn="cosine",
    )

    return vector_store


def load_vectorestore_from_Mongo(embeddings):

    # create mongodb vectore store
    collection = get_connection()

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
    )

    return vector_store


def create_vectorstore_from_FAISS(pdf_path, vectorstore_path, embeddings):

    chunked_docs = pdf_splitter(pdf_path)

    # 단계 3,4: 임베딩 & 벡터 스토어 생성
    # FAISS 사용
    vectorstore = FAISS.from_documents(documents=chunked_docs, embedding=embeddings)

    vectorstore.save_local(vectorstore_path)

    return vectorstore


def load_vectorstore_from_FAISS(vectorstore_path, embeddings):
    """Load an existing vectorstore."""
    return FAISS.load_local(
        vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True
    )


def create_conversaion_memory(memory_key, output_key):
    # 메모리 컴포넌트 생성
    memory = ConversationBufferWindowMemory(
        return_messages=True,
        memory_key=memory_key,
        output_key=output_key,
    )

    return memory
