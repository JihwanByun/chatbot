import os

import torch
from config import LOCAL_MODEL_PATH, MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Sembot:
    """
    local model로 RAG 기반 LLM 채팅서비스
    """

    def __init__(
        self,
        local_model_path=LOCAL_MODEL_PATH,
        model_name=MODEL_NAME,
    ):
        self.tokenizer, self.model = self._get_tokenizer_model(
            local_model_path, model_name
        )
        self.streamer = self._get_streamer(self.tokenizer)
        self.terminators = self._get_terminator(self.tokenizer)

    def get_model(self):
        return self.model

    def get_streamer(self):
        return self.streamer

    def _create_model_tokenizer(self, model_name: str, save_path: str):
        """
        model과 tokenizer를 로컬에 저장합니다.

        Args:ㅏ
            model_name (str): 불러올 모델의 이름
            save_path (str): 불러온 모델과 토큰나이저를 저장할 로컬 경로
        """

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    def inference(self, prompt):

        # 프롬프트 토크나이징
        input_ids = self._make_input_ids(prompt)

        # 생성에 필요한 인자 설정
        generation_kwargs = self._make_generation_kwargs(input_ids)

        return generation_kwargs


if __name__ == "__main__":
    from threading import Thread
    from prepare_sembot import SembotDocs

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

    sembot = Sembot()
    model = sembot.get_model()
    streamer = sembot.get_streamer()

    generation_kwargs = sembot.inference(prompt)

    # Thread 객체를 사용하기 때문에 동시성 이슈를 신경써야함
    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    thread.start()

    for new_text in streamer:

        # print가 아니라 프론트로 전송하는 걸로 바꾸면 될 듯
        print(new_text, end="", flush=True)

    thread.join()
