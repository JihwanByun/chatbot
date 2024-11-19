import asyncio
import os
from summarizer import PDFSummarizer
from sambot.utils.llm_factory import llm_factory

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 사용 예시
async def main():
    pdf_path = "C:\\Users\\SSAFY\\Desktop\\SSAFY\\S11P31S102\\AI\\sambot\\1.make_vector_store\\kisa_pdf\\3_06_출장규칙(240214).pdf"

    llm=llm_factory.get_llm("gpt")
    summarizer = PDFSummarizer(llm)
    result = await summarizer.summarize(pdf_path, use_parallel=True, extractor="pdfplumber")
    # result = await summarizer.summarize(pdf_path, use_parallel=True, extractor="pypdf")

    # 결과를 다룰 때
    if result["success"]:
        print("Summary:", result["summary"])
    else:
        print("Summarization failed.")

# 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(main())