from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from mongo_client import MongoDBClient

def file_splitter(pdf_file, mongo_data):
    """
    PDF를 splitter로 작은 청크로 분리합니다.

    Args:
        pdf_file: pdf_file을 받습니다.

    Returns:
        청크로 분리된 pdf 파일 객체 리스트
    """

    all_documents = []

    # 텍스트를 작은 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 각 청크의 최대 문자 수
        chunk_overlap=32,  # 청크 간 중복되는 문자 수
    )

    # board
    # gpu 서버에서는 몽고 디비 포트로 접근이 안됨
    # client = MongoDBClient()

    # regulations = client.find_regulations()

    for regulation in mongo_data:       
        documents = make_data_for_vectorDB(regulation)

        chunked_documents = text_splitter.split_documents(documents)
        all_documents.extend(chunked_documents)

    # pdf
    # 현재 pdf는 하나만 가능하도록 함
    if(pdf_file["content"]):
        documents = text_splitter.create_documents([pdf_file["content"]])
        documents[0].metadata["file_name"] = pdf_file["title"]
        documents[0].metadata["level"] = pdf_file["level"]

        chunked_documents = text_splitter.split_documents(documents)
        all_documents.extend(chunked_documents)

    return all_documents


def make_data_for_vectorDB(regulation):

    file_name = regulation["title"]

    pages = regulation["itemList"]

    page_title = ""
    all_documents = []

    for index, page in enumerate(pages, 1):
        page_title = f"제{index}장({page['content']})"

        for article_index, article in enumerate(page["itemList"], 1):
            content = ""

            article_title = article['title'] if article['title'] else ""
            
            data_source = (
                f"{file_name} {page_title} 제{article_index}조({article_title})"
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
                metadata={"file_name": data_source, "level": regulation["level"]},
                page_content=content,
            )

            all_documents.append(document)

    return all_documents


def create_vector_store(vectorstore_path, embeddings, pdf_file, mongo_data):
    chunked_docs = file_splitter(pdf_file, mongo_data)

    vectorstore = FAISS.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )

    print("vectorstore 생성 중....")
    vectorstore.save_local(vectorstore_path)
    print("vectorstore 생성 완료....")

    return vectorstore