# MODEL
PROMPT = """You are a knowledgeable AI assistant trained to answer questions about company regulations. Respond directly and concisely, using the information from company policies. Answer as follows: '사내 규정에 따르면…' and provide a clear response based on the retrieved content.

당신은 회사 규정에 대해 질문을 명확하고 직접적으로 답변하는 유능한 AI 어시스턴트입니다. 답변은 '사내 규정에 따르면…'의 형식으로 시작하여, 검색된 내용에 따라 명확한 답을 제공하세요."""

# Paths
PDF_DIR_PATH = r"../kisa_pdf/"
VECTORSTORE_PATH = r"./vector_store_path"

# MODEL_PATH = r"llama3.1"
MODEL_PATH = r"gpt-4o-mini"

VECTORSTORE_PATH = r"./vector_store_path_jhgan"
EMBEDDINGS_MODEL_NAME = "jhgan/ko-sroberta-nli"

# VECTORSTORE_PATH = r"./vector_store_path_intfloat"
# EMBEDDINGS_MODEL_NAME = "intfloat/multilingual-e5-large"

# Embeddings Configuration
EMBEDDINGS_MODEL_KWARGS = {"device": "cuda"}
EMBEDDINGS_ENCODE_KWARGS = {"normalize_embeddings": True}

MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
LOCAL_MODEL_PATH = f"./local_model_{MODEL_NAME}"

# MODEL_NAME = "NCSOFT/Llama-VARCO-8B-Instruct"
# LOCAL_MODEL_PATH = f"./local_model_{MODEL_NAME}"
