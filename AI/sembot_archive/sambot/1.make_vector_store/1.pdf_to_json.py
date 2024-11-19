import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 설정 부분
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
PDF_PATH = os.path.join(BASE_DIR, '1.make_vector_store\kisa_pdf')
JSON_PATH = os.path.join(BASE_DIR, '1.make_vector_store\kisa.json')

# PDF 파일명들을 로드하는 함수
def load_pdf_files(pdf_path):
    """주어진 디렉토리에서 모든 PDF 파일들을 로드합니다."""
    pdf_files = []
    for root, dirs, files in os.walk(pdf_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(file)
    return pdf_files

# PDF를 JSON으로 변환하는 함수
def pdf_to_json(pdf_path, json_path):
    pdf_files = load_pdf_files(pdf_path)
    all_documents = []

    for pdf_file in pdf_files:
        loader = PDFPlumberLoader(os.path.join(PDF_PATH, pdf_file))
        documents = loader.load()  # PDF 문서를 로드합니다

        # 텍스트를 작은 청크로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # 각 청크의 최대 문자 수
            chunk_overlap=64  # 청크 간 중복되는 문자 수
        )
        chunked_documents = text_splitter.split_documents(documents)
        print(f"페이지수: {len(documents)} 청크수: {len(chunked_documents)} 파일명: {pdf_file}")
        all_documents.extend(chunked_documents)

    # 데이터 검수를 위해 JSON 파일로 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json_data = [
            {
                "metadata": {
                    **doc.metadata,  # 기존 메타데이터 포함
                    "file_name": os.path.basename(doc.metadata['source'])  # 파일명만 추가
                },
                "page_content": doc.page_content
            }
            for doc in all_documents
        ]
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"모든 PDF 파일이 {json_path}로 JSON 변환되었습니다.")

def main():
    print("PDF를 JSON으로 변환합니다...")
    pdf_to_json(PDF_PATH, JSON_PATH)
    print("PDF 파일이 JSON 파일로 변환되었습니다.")

if __name__ == "__main__":
    main()