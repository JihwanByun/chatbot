import json
import os
from pathlib import Path

from config import (
    EMBEDDINGS_ENCODE_KWARGS,
    EMBEDDINGS_MODEL_KWARGS,
    EMBEDDINGS_MODEL_NAME,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


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
def load_files(file_path: str):
    """
    주어진 디렉토리에서 모든 PDF 파일들을 로드합니다.

    Args:
        file_path (str): file 디렉토리 경로
        file_type (str): file type

    Returns:
        pdf 파일 경로 리스트

    """

    data_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            data_files.append(os.path.join(root, file))

    return data_files


def file_splitter(file_dir_path: str):
    """
    PDF를 splitter로 작은 청크로 분리합니다.

    Args:
        file_dir_path (str): pdf 파일이 있는 디렉토리 경로

    Returns:
        청크로 분리된 pdf 파일 객체 리스트
    """
    files = load_files(file_dir_path)

    all_documents = []

    # 텍스트를 작은 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 각 청크의 최대 문자 수
        chunk_overlap=32,  # 청크 간 중복되는 문자 수
    )

    for file in files:
        if file.endswith("pdf"):
            loader = PDFPlumberLoader(file)
            documents = loader.load()  # PDF 문서를 로드합니다

        elif file.endswith("json"):
            documents = make_data_for_vectorDB(file)

        chunked_documents = text_splitter.split_documents(documents)
        all_documents.extend(chunked_documents)

    return all_documents


def make_data_for_vectorDB(json_file):

    json_data = json.loads(Path(json_file).read_text(encoding="utf-8"))

    file_name = json_file.split("/")[-1].split(".")[0]

    pages = json_data["itemList"]

    page_title = ""
    all_documents = []

    for index, page in enumerate(pages, 1):
        page_title = f"제{index}장({page['content']})"

        for article_index, article in enumerate(page["itemList"], 1):
            content = ""

            data_source = (
                f"{file_name} {page_title} 제{article_index}조({article['title']})"
            )

            # article의 content가 있으면 추가
            if article.get("content"):
                content += article["content"] + "\n"

            # 항이 있으면 추가
            if article.get("itemList"):
                for para_index, paragraph in enumerate(article["itemList"], 1):
                    content += f"제{para_index}항 {paragraph['content']}\n"

                    # 호가 있으면 추가
                    if paragraph.get("itemList"):
                        for subPara_index, sub_paragraph in enumerate(
                            paragraph["itemList"], 1
                        ):
                            content += (
                                f"제{subPara_index}호 {sub_paragraph['content']}\n"
                            )

            document = Document(
                metadata={"file_name": data_source, "level": json_data["level"]},
                page_content=content,
            )

            all_documents.append(document)

    return all_documents
