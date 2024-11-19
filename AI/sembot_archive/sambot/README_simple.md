<!-- conda 터미널에서 아래대로 따라하시면 됩니다. -->

```bash
conda create -n py310 python=3.10
conda activate py310
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
python cuda_check.py
pip install transformers langchain huggingface_hub langchain-community langchain-huggingface langchain_openai fastapi uvicorn pypdf accelerate python-dotenv python-multipart
conda install -c conda-forge faiss-gpu
python app.py
```