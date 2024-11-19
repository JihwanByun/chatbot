from tempfile import NamedTemporaryFile
from typing import Dict, Any, List, Literal
from pathlib import Path

from fastapi import UploadFile, File
from pypdf import PdfReader
import pdfplumber
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from operator import itemgetter
import logging
import os

from concurrent.futures import ThreadPoolExecutor

from sambot.utils.timer import timeit, async_timeit

from sambot.summarize.summarizer_prompts import MAP_PROMPT_TEXT, COMBINE_PROMPT_TEXT

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

class PDFSummarizer:
    def __init__(self, llm, chunk_size: int = 2048, chunk_overlap: int = 256):
        """
        PDF 요약기 초기화

        Args:
            llm: LangChain 호환 LLM 인스턴스
            chunk_size: 텍스트 분할 크기
            chunk_overlap: 텍스트 분할 오버랩 크기
        """
        logging.basicConfig(level=logging.INFO)

        self.llm = llm
        self.text_splitter = self._create_text_splitter(chunk_size, chunk_overlap)
        self.map_chain, self.combine_chain = self._create_chains()
        self.MAX_TOKEN = 8096

    @staticmethod
    def _create_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """텍스트 분할기 생성"""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def _create_chains(self) -> tuple[RunnableSequence, RunnableSequence]:
        """맵 체인과 결합 체인 생성"""
        # 맵 프롬프트 (각 청크 요약용)
        map_prompt = PromptTemplate.from_template(MAP_PROMPT_TEXT)

        # 결합 프롬프트 (최종 요약용)
        combine_prompt = PromptTemplate.from_template(COMBINE_PROMPT_TEXT)

        # 맵 체인 생성
        map_chain = (
                {"text": itemgetter("text")}
                | map_prompt
                | self.llm
                | StrOutputParser()
        )

        # 결합 체인 생성
        combine_chain = (
                {"text": itemgetter("text")}
                | combine_prompt
                | self.llm
                | StrOutputParser()
        )

        return map_chain, combine_chain

    @timeit
    def _summarize_document(self, doc: Document) -> str:
        """단일 문서 요약"""
        return self.map_chain.invoke({"text": doc.page_content})

    @async_timeit
    async def _map_reduce(self, docs: list[Document], use_parallel: bool = False) -> str:
        """맵-리듀스 실행 (병렬 여부 선택)"""
        summaries = []
        total_docs = len(docs)

        if use_parallel:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._summarize_document, doc): idx for idx, doc in enumerate(docs)}
                for future in futures:
                    idx = futures[future]
                    summaries.append(future.result())
                    print(f"병렬 처리 - 문서 {idx + 1} / {total_docs} 처리 완료")
        else:
            for idx, doc in enumerate(docs):
                summaries.append(self._summarize_document(doc))
                print(f"비병렬 처리 - 문서 {idx + 1} / {total_docs} 처리 완료")

        # 요청 max token 수를 벋어나는지 검증 후 축약
        if self._should_collapse(summaries):
            summaries = self._collapse_summaries(summaries)

        logging.info(f"최종 입력 토큰 수: {self._length_function(summaries)}")
        # 요약 결합
        combined_summary = self.combine_chain.invoke({"text": "\n\n".join(summaries)})
        return combined_summary


    @staticmethod
    def _extract_text_with_pdfplumber(pdf_path: str | Path) -> str:
        """pdfplumber를 사용하여 PDF에서 텍스트 추출"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + " "
        return text

    @staticmethod
    def _extract_text_with_pypdf(pdf_path: str | Path) -> str:
        """PDF에서 텍스트 추출"""
        try:
            pdf_reader = PdfReader(pdf_path)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            raise ValueError(f"PDF 텍스트 추출 중 오류 발생: {str(e)}")

    @timeit
    def _extract_text_from_pdf(
            self,
            pdf_path: str | Path,
            extractor: Literal["pypdf", "pdfplumber"] = "pdfplumber",
    ) -> str:
        """PDF에서 텍스트, 이미지, 테이블 추출"""
        try:
            # 텍스트 추출
            if extractor == "pypdf":
                text = self._extract_text_with_pypdf(pdf_path)
            elif extractor == "pdfplumber":
                text = self._extract_text_with_pdfplumber(pdf_path)
            else:
                raise ValueError(f"지원하지 않는 추출기: {extractor}")

            return text
        except Exception as e:
            raise ValueError(f"PDF 콘텐츠 추출 중 오류 발생: {str(e)}")

    async def _extract_text_from_upload_file(
            self,
            file: UploadFile,
            extractor: Literal["pypdf", "pdfplumber"] = "pdfplumber",
    ) -> Dict[str, Any]:
        """UploadFile에서 텍스트, 이미지, 테이블 추출"""
        try:
            # 임시 파일로 저장
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file.flush()

                # 선택된 추출기로 PDF 읽기
                result = self._extract_text_from_pdf(
                    tmp_file.name,
                    extractor
                )

            # 임시 파일 삭제
            os.unlink(tmp_file.name)
            return result
        except Exception as e:
            raise ValueError(f"업로드된 PDF 파일 처리 중 오류 발생: {str(e)}")

    @timeit
    def _collapse_summaries(self, contents: List[str]) -> List[str]:
        """
        요약된 내용들을 재귀적으로 결합

        Args:
            contents: 요약된 내용들의 리스트

        Returns:
            최종 결합된 요약 리스트
        """

        # 결합된 요약들을 저장할 리스트
        collapsed_summaries = []
        current_content = []
        num_current_tokens = 0

        for content in contents:
            # 현재 콘텐츠의 토큰 수
            num_content_tokens = self.llm.get_num_tokens(content)

            # 현재 그룹에 추가할 경우 최대 길이를 초과하는지 확인
            if num_current_tokens + num_content_tokens > self.MAX_TOKEN and current_content:
                # 현재까지의 내용을 결합하여 요약
                combined_text = "\n\n".join(current_content)
                summary = self.combine_chain.invoke({"text": combined_text})
                collapsed_summaries.append(summary)

                # 새로운 그룹 시작
                current_content = [content]
                num_current_tokens = num_content_tokens
            else:
                # 현재 그룹에 추가
                current_content.append(content)
                num_current_tokens += num_content_tokens

        # 마지막 그룹 처리
        if current_content:
            combined_text = "\n\n".join(current_content)
            summary = self.combine_chain.invoke({"text": combined_text})
            collapsed_summaries.append(summary)

        # 재귀적으로 처리
        if self._should_collapse(collapsed_summaries):
            return self._collapse_summaries(collapsed_summaries)

        return collapsed_summaries

    def _should_collapse(self, contents: List[str]):
        num_tokens = self._length_function(contents)
        logging.info(f"총 입력 토큰 수: {num_tokens}")
        return num_tokens > self.MAX_TOKEN

    def _length_function(self, contents: List[str]) -> int:
        return sum(self.llm.get_num_tokens(content) for content in contents)

    @async_timeit
    async def summarize(
            self,
            pdf_info: str | Path | UploadFile,
            use_parallel: bool = False,
            extractor: Literal["pypdf", "pdfplumber"] = "pdfplumber") -> Dict[str, Any]:
        """
        PDF 파일을 요약

        Args:
            pdf_path: PDF 파일 경로
            use_parallel: 병렬 처리 여부

        Returns:
            요약 결과를 포함한 딕셔너리
        """
        try:
            # PDF 텍스트 추출
            text = ""
            if isinstance(pdf_info, (str, Path)):
                logging.info("input 형식 : path")
                text = self._extract_text_from_pdf(pdf_info,extractor)
            else:
                logging.info("input 형식 : file")
                text = await self._extract_text_from_upload_file(pdf_info,extractor)

            # 텍스트 분할
            docs = self.text_splitter.create_documents([text])

            # 맵-리듀스 요약 실행
            summary = await self._map_reduce(docs, use_parallel)

            return {
                "success": True,
                "summary": summary,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "summary": None,
                "error": str(e)
            }
