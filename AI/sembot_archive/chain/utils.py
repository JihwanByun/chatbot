from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import (
    EMBEDDINGS_MODEL_NAME,
    EMBEDDINGS_MODEL_KWARGS,
    EMBEDDINGS_ENCODE_KWARGS,
)


def summarize_documents(documents):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in documents)


def print_retrieved_docs(docs):
    """Retrieved documents를 출력하는 함수"""
    print("\n=== Retrieved Documents ===")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"{doc.metadata['source']} |  page: {doc.metadata['page']}")
        print(doc.page_content)
        print("-" * 50)

    print("\n\nAnser: \n")
    return docs


def format_docs_with_print(docs):
    """문서를 출력하고 포맷팅하는 함수"""
    print_retrieved_docs(docs)
    return summarize_documents(docs)


def get_embeddings():
    """Get embeddings for the vectorstore."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=EMBEDDINGS_MODEL_KWARGS,
        encode_kwargs=EMBEDDINGS_ENCODE_KWARGS,
    )


def invoke_chain_with_memory(question: str, answer: str, memory):

    # 메모리에 대화 저장
    memory.save_context({"question": question}, {"answer": answer})

def create_prompt():
    # 새로운 프롬프트 템플릿 생성 (기존 rlm/rag-prompt를 기반으로 수정)
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise. Answer in Korean.'''),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ])

    return prompt