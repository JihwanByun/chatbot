import json
import os

import torch
from config import LOCAL_MODEL_PATH, MODEL_NAME, PDF_DIR_PATH, PROMPT, VECTORSTORE_PATH
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from utils import get_embeddings
from vectorstore_manager import get_vectorestore_by_FAISS


class Sembot:
    """
    local model로 RAG 기반 LLM 채팅서비스
    """

    def __init__(
        self,
        local_model_path=LOCAL_MODEL_PATH,
        model_name=MODEL_NAME,
        vectorstore_path=VECTORSTORE_PATH,
        pdf_dir_path=PDF_DIR_PATH,
    ):
        self.tokenizer, self.model = self._get_tokenizer_model(
            local_model_path, model_name
        )
        self.streamer = self._get_streamer(self.tokenizer)
        self.terminators = self._get_terminator(self.tokenizer)
        self.vector_store = get_vectorestore_by_FAISS(vectorstore_path, pdf_dir_path)
        self.retriever = self.vector_store.as_retriever()

    def get_model(self):
        return self.model

    def get_streamer(self):
        return self.streamer

    def _init_prompt(self):
        prompt = []

        prompt.append({"role": "system", "content": PROMPT})

        return prompt

    def _create_model_tokenizer(self, model_name: str, save_path: str):
        """
        model과 tokenizer를 로컬에 저장합니다.

        Args:ㅏ
            model_name (str): 불러올 모델의 이름
            save_path (str): 불러온 모델과 토큰나이저를 저장할 로컬 경로
        """

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 모델을 로컬 경로에 저장 (예: ./models/gpt2)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("모델 다운로드 완료")

    def _load_model_tokenizer(self, model_path: str):
        """
        로걸에 저장한 모델과 토큰나이저를 불러옵니다.

        Args:
            model_path (str): 저장한 모델과 토큰나이저 경로

        Returns:
            tokenizer, model: 로컬 경로에서 불러온 토큰나이저와 모델
        """

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model.eval()

        return tokenizer, model

    def _get_tokenizer_model(self, model_path: str, model_name: str):
        """
        tokenizer와 model을 local 경로에서 가져옵니다.

        Args:
            model_path (str): 저장한 모델과 토큰나이저 경로
            model_name (str): 불러올 모델의 이름

        Returns:
            _type_: _description_
        """
        if os.path.exists(model_path):
            print("Loading existing Local Model...")
            tokenizer, llm = self._load_model_tokenizer(model_path)
        else:
            print("Local Model not found. Creating a new one...")
            self._create_model_tokenizer(model_name, model_path)
            tokenizer, llm = self._load_model_tokenizer(model_path)

        return tokenizer, llm

    def _get_streamer(self, tokenizer):
        """
        모델이 생성하는 text를 iterator streamer 객체로 만들 수 있는 객체를 만듭니다.

        Args:
            tokenizer (_type_): 토큰나이저

        Returns:
            TextIteratorStreamer: TextIteratorStreamer 객체
        """
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        return streamer

    def _get_terminator(self, tokenizer):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        return terminators

    def _make_generation_kwargs(
        self,
        input_ids,
        max_new_toknes=2048,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1,
    ):
        """
        모델로 답변을 생성할 때 필요한 인자를 정의합니다.

        Args:
            input_ids (_type_): 토큰나이징 된 입력
            streamer (_type_): 스트리밍에 필요한 streamer 객체
            max_new_toknes (int, optional): 최대 생성 토큰 수. Defaults to 2048.
            temperature (float, optional): 확률분포를 조정하는 매개변수, 모델의 창의력(0 ~ 1). Defaults to 0.1.
            top_p (float, optional): 확률분포 내에서 선택할 단어의 범위를 결정하는 매개변수. Defaults to 0.9.
            repetition_penalty (float, optional): 모델이 생성하는 답변의 반복을 통제 (1.0 <= x). Defaults to 1.1.

        Returns:
            dict: 생성에 필요한 인자 dict
        """
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": self.streamer,
            "max_new_tokens": max_new_toknes,
            "eos_token_id": self.terminators,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        return generation_kwargs

    def _make_input_ids(self, prompt: list):
        """
        tokenizer로 질문을 토큰나이징 합니다.

        Args:
            prompt list: 시스템, 이전 대화, 연관 문서, 질문 프롬프트 리스트

        Returns:
            _type_: 질문에 대해서 채팅 템플릿으로 토크나이징 된 값
        """

        input_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        return input_ids

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

    def _make_prompt(self, request):

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

    def chain(self, request):
        # 프롬프트 생성
        prompt = self._make_prompt(request)

        # 프롬프트 토크나이징
        input_ids = self._make_input_ids(prompt)

        # 생성에 필요한 인자 설정
        generation_kwargs = self._make_generation_kwargs(input_ids)

        return generation_kwargs

    def docs_invoke(self, question):
        return self.retriever.invoke(question)


if __name__ == "__main__":
    from threading import Thread

    sembot = Sembot()

    model = sembot.get_model()
    streamer = sembot.get_streamer()

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

    generation_kwargs = sembot.chain(input_data)

    # Thread 객체를 사용하기 때문에 동시성 이슈를 신경써야함
    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    thread.start()

    for new_text in streamer:

        # print가 아니라 프론트로 전송하는 걸로 바꾸면 될 듯
        print(new_text, end="", flush=True)

    thread.join()
