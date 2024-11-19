# batch.py

import torch
from torch.nn.utils.rnn import pad_sequence
import asyncio
from prepare import load_prompt, format_docs, memory_retriever, filter_docs
from streamer import BatchTextStreamer

def make_generation_kwargs(input_ids, eos_token_id, streamer):
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": 400,
        "eos_token_id": eos_token_id,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.9,
        "streamer": streamer
    }
    return generation_kwargs


def generate_model(model, generation_kwargs):
    with torch.no_grad():
        model.generate(**generation_kwargs)
    # 메모리 정리
    torch.cuda.empty_cache()


# 배치 처리 백그라운드 태스크
async def process_queue(request_queue, batch_size, batch_lock, retriever_lock, event_loop, tokenizer, model, eos_token_id, embeddings, retriever):
    while True:
        # 최신 retriever 가져오기
        async with retriever_lock:
            current_retriever = retriever
            
        async with batch_lock:
            batch = []
            response_queues = []


            # 첫 번째 요청을 기다립니다.
            try:
                request = await request_queue.get()
                input_data = {'question': request['question'], 'memory': request['memory'], 'level': request['level']}
                batch.append(input_data)
                response_queues.append(request['response_queue'])
            except Exception as e:
                continue  # 예외 처리

            # 추가 요청을 배치 사이즈만큼 수집합니다.
            while len(batch) < batch_size:
                try:
                    # 대기하지 않고 요청이 있으면 즉시 가져옵니다.
                    request = request_queue.get_nowait()
                    input_data = {'question': request['question'], 'memory': request['memory'], 'level': request['level']}
                    batch.append(input_data)
                    response_queues.append(request['response_queue'])
                except asyncio.QueueEmpty:
                    break  # 대기열이 비어 있으면 종료

            if batch:
                # 배치된 요청들에 대해 메시지 생성
                messages = []
                for input_data in batch:
                    message = []
                    message.append({"role": "system", "content": load_prompt()})
                    for qda in input_data["memory"]:
                        message.append({"role": "user", "content": qda.get("question", "")})
                        message.append({"role": "assistant", "content": qda.get("answer", "")})

                    # 사용자 레벨에 맞는 문서 필터링
                    filtered_docs = filter_docs(current_retriever.invoke(input_data['question']), input_data["level"])
                    
                    message.append({"role": "user", "content": f"{format_docs(memory_retriever(input_data, embeddings))}\n\n {format_docs(filtered_docs)}\n\n {input_data['question']}"})
                    messages.append(message)

                # 배치 입력 생성
                input_ids = [
                    tokenizer.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(model.device).squeeze(0)
                    for msg in messages
                ]
                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

                # 커스텀 스트리머 생성
                streamer = BatchTextStreamer(tokenizer, response_queues, event_loop, eos_token_id=eos_token_id, skip_prompt=True)

                # 모델 배치 추론
                generation_kwargs = make_generation_kwargs(input_ids, eos_token_id, streamer)

                # 모델 생성을 비동기로 실행하고 완료될 때까지 기다립니다.
                await asyncio.to_thread(generate_model, model, generation_kwargs)
                # batch_lock은 여기까지 유지됩니다.