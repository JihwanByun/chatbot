import json

from config import PDF_DIR_PATH, PROMPT, VECTORSTORE_PATH
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from utils import get_embeddings
from vectorstore_manager import get_vectorestore_by_FAISS


class SembotDocs:
    """
    local model로 RAG 기반 LLM 채팅서비스
    """

    def __init__(
        self,
        vectorstore_path=VECTORSTORE_PATH,
        pdf_dir_path=PDF_DIR_PATH,
    ):
        self.vector_store = get_vectorestore_by_FAISS(vectorstore_path, pdf_dir_path)
        self.retriever = self.vector_store.as_retriever()

    def _init_prompt(self):
        prompt = []

        prompt.append({"role": "system", "content": PROMPT})

        return prompt

    def _format_memory(self, request, prompt):
        """
        question, document, answer로 이루어진 대화 쌍 리스트에서
        질문과 답변만 추출해서 문자열로 만듭니다.

        Args:
            memory (_type_): question, document, answer로 이루어진 대화 쌍 리스트

        Returns:
            str: 질문과 답변 문자열
        """

        for i, qda in enumerate(request["memory"], start=1):
            # qda에서 question 추출
            question = qda.get("question", "")
            if question:
                prompt.append({"role": "user", "content": question})

            # qda에서 answer 추출
            answer = qda.get("answer", "")
            if answer:
                prompt.append({"role": "assistant", "content": answer})

        return prompt

    def _format_docs(self, docs: list):
        """
        document 객체 리스트에서
        각 document에 페이지 정보, 문서 내용을 추출해서 문자열로 만듭니다.

        Args:
            docs (_type_): document 객체 리스트

        Returns:
            str: 각 document에 페이지 정보, 문서 내용을 추출한 문자열
        """

        context = ""
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content
            source = doc.metadata.get("source", "Unknown source")
            context += f"Document (Source: {source}):\n{content}\n\n"

        return context.strip()

    def _memory_retriever(self, request):
        """
        question, document, answer로 이루어진 대화 쌍 리스트에 문서 중
        현재 질문과 가장 유사한 document를 반환합니다.

        Args:
            request (_type_): question, document, answer로 이루어진 대화 쌍 리스트

        Returns:
            str: 현재 질문과 가장 유사한 document 객체 리스트
        """
        memory = request["memory"]
        question = request["question"]
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
        memory_vectorstore = FAISS.from_documents(docs, get_embeddings())
        memory_retriever = memory_vectorstore.as_retriever()

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

    def _format_related_docs(self, request):
        """
        사용자의 입력(대화 기록, 현재 질문)에서
        대화 기록에서 현재 질문과 유사한 문서를 추출합니다.

        Args:
            inputs (_type_): 사용자의 입력(대화 기록, 현재 질문)

        Returns:
            str: 연관된 문서 문자열
        """
        retreieved_docs = self._memory_retriever(request)

        formated_docs = self._format_docs(retreieved_docs)

        return formated_docs

    def make_prompt(self, request):

        prompt = self._init_prompt()

        prompt = self._format_memory(request, prompt)

        formated_memory_docs = self._format_related_docs(request)

        formated_docs = self._format_docs(self.retriever.invoke(request["question"]))

        prompt.append(
            {
                "role": "user",
                "content": f"{formated_memory_docs}\n\n {formated_docs}\n\n {request['question']}",
            }
        )

        return prompt

    def docs_invoke(self, question):
        return self.retriever.invoke(question)


if __name__ == "__main__":
    input_data = {
        "memory": [
            {
                "question": "유류비 청구는 어떻게 할 수 있나요? 제가 3박 4일로 부산에 갓다오는데 비용은 대충 어떻게 계산하죠?",
                "doc": {
                    "metadata": {"source": "유류비_소스.pdf"},
                    "page_content": "유류비 청구와 관련된 내용",
                },
                "answer": "안녕하세요! 부산으로 출장가는 경우, 유류비 청구는 다음과 같은 절차를 따르면 됩니다.",
            },
            {
                "question": "연차는 어떻게 신청할 수 있나요?",
                "doc": {
                    "metadata": {"source": "연차_소스.pdf"},
                    "page_content": "연차와 관련된 내용",
                },
                "answer": "안녕하세요! 연차는 다음과 같은 절차를 따르면 신청할 수 있습니다.",
            },
        ],
        "question": "제가 아까 무슨 질문 했죠? 기억나시나요??",
    }

    pre_sembot = SembotDocs()

    prompt = pre_sembot.make_prompt(input_data)

    docs = pre_sembot.docs_invoke(input_data)
