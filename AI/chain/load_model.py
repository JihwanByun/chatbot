from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def create_model(model_name, model_path):

    # 모델과 토크나이저 다운로드 후 로컬에 저장
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 모델을 로컬 경로에 저장 (예: ./models/gpt2)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print("모델 다운로드 완료")


def load_model(model_path):
    # 토크나이저와 모델 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    return tokenizer, model
