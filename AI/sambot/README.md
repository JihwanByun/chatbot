<!-- 설명서.md -->

### 1. env파일 만들기 (모델 폴더 절대 경로 지정)
```txt
BASE_DIR='디렉토리 저장된 절대경로'
OPENAI_API_KEY='필요하면 빌려드립니다.'
```


### 2. Conda 환경 생성 (venv도 가능하나 비추)
**Python 3.10버전으로 설치**
```bash
conda create -n py310 python=3.10
conda activate py310
```


### 3. PyTorch conda 설치 (2GB 다운로드)
[**PyTorch 로컬 설치 가이드**](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
python cuda_check.py
```


### 4. pip install
```bash
pip install transformers langchain huggingface_hub langchain-community langchain-huggingface langchain_openai fastapi uvicorn pypdf accelerate python-dotenv python-multipart
```
- `transformers`: 허깅페이스 모델 로드
- `langchain`: LangChain 프레임워크
- `huggingface_hub`: Hugging Face CLI
- `langchain-community`: LangChain 커뮤니티에서 제공하는 추가 기능
- `langchain-huggingface`: Hugging Face와의 통합을 위한 LangChain 확장
- `langchain-openai`: gpt llm 로드
- `fastapi`: 고성능 웹 프레임워크
- `uvicorn`: ASGI 서버
- `pypdf`: PDF 파일 처리 라이브러리
- `accelerate`: 하드웨어 가속기
- `python-dotenv` : env파일 로드


### 5. faiss-gpu는 pip install로 불가
```bash
conda install -c conda-forge faiss-gpu
```
- `faiss-gpu`: GPU 가속 벡터 검색 라이브러리


### 6. 벡터 DB만들기 (vector_store_path 있으면 생략 가능)
**PDF -> json -> VectorDB**
```bash
python 1.make_vector_store/1.pdf_to_json.py
python 1.make_vector_store/2.json_to_vector.py
python 1.make_vector_store/3.test_vector_store.py
```


### 7. LLM모델 테스트
```bash
python 2.make_model/1.test_only_model.py
python 2.make_model/2.test_vector_model.py
```


### 8. FastAPI를 이용한 모델 서비스
**서버 실행 후 client.html로 서비스 확인**
```bash
python app.py
```