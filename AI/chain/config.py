import os

# MongoDB
MONGODB_CLUSTER_URI = "mongodb://localhost:27017"
MONGODB_DB_NAME = "vector_db"

# Paths
PDF_PATH = r"../kisa_pdf/"
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

# BASE_DIR = os.getenv('BASE_DIR')
# LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "local_model")
LOCAL_MODEL_PATH = r"./local_model"
MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
