import os

from config import (
    EMBEDDINGS_ENCODE_KWARGS,
    EMBEDDINGS_MODEL_KWARGS,
    EMBEDDINGS_MODEL_NAME,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    벡터 DB에 사용할 임베딩 모델을 가져옵니다.

    Returns:
        HuggingFaceEmbeddings

    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=EMBEDDINGS_MODEL_KWARGS,
        encode_kwargs=EMBEDDINGS_ENCODE_KWARGS,
    )


# PDF 파일명들을 로드하는 함수
def load_pdf_files(pdf_path: str):
    """
    주어진 디렉토리에서 모든 PDF 파일들을 로드합니다.

    Args:
        pdf_path (str): pdf 디렉토리 경로

    Returns:
        pdf 파일 경로 리스트

    """
    pdf_files = []
    for root, dirs, files in os.walk(pdf_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    return pdf_files


def pdf_splitter(pdf_dir_path: str):
    """
    PDF를 splitter로 작은 청크로 분리합니다.

    Args:
        pdf_path (str): pdf 파일이 있는 디렉토리 경로

    Returns:
        청크로 분리된 pdf 파일 객체 리스트
    """
    pdf_files = load_pdf_files(pdf_dir_path)
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
