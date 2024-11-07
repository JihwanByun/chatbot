import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import os
import dotenv

'''
테스트를 위한 llm 모델생성기
지우셔도 됩니다.
'''

# 현재 할당된 메모리를 해제
torch.cuda.empty_cache()

dotenv.load_dotenv()

BASE_DIR = os.getenv('BASE_DIR')
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vector_store_path")
MODEL_PATH = os.path.join(BASE_DIR, "local_model")

class LLMFactory:
    _instance = None

    def __new__(cls):
        """LLMManager의 인스턴스를 싱글톤으로 생성합니다."""
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
            cls._instance._initialize_llms()  # LLM 초기화
        return cls._instance

    def _initialize_llms(self):
        """여러 LLM 인스턴스를 초기화합니다."""
        self.llms = {
            # "local": self._create_llm(MODEL_PATH),
            "gpt" : self._create_llm_gpt()
        }

    def _create_llm(self, model_path):
        """주어진 모델 경로에 대해 LLM 인스턴스를 생성합니다."""
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=0,
            top_p=0.9,
            return_full_text=False,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )

        return HuggingFacePipeline(pipeline=pipe)

    def _create_llm_gpt(self):
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        return llm

    def get_llm(self, name):
        """주어진 이름에 적합한 LLM을 반환합니다."""
        return self.llms.get(name)

# LLMFactory를 통해 싱글톤 LLM 인스턴스를 가져오기
llm_factory = LLMFactory()  # 싱글톤 인스턴스 생성
