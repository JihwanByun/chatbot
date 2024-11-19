import os
import torch

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}") # gpu번호, none이어도 상관없음
print("CUDA :", torch.cuda.is_available())  # True가 나오면 CUDA 사용 가능
print("Ver :", torch.version.cuda)  # 설치된 PyTorch에서 사용하는 CUDA 버전
print("GPU :",torch.cuda.get_device_name(0))  # 사용하는 GPU 이름 출력되어야 함