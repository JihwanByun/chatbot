# streamer.py

import asyncio
from transformers.generation.streamers import BaseStreamer


class BatchTextStreamer(BaseStreamer):
    def __init__(self, tokenizer, queues, event_loop, eos_token_id, skip_prompt=False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.queues = queues  # 각 요청의 응답 큐 리스트
        self.event_loop = event_loop
        self.eos_token_id = eos_token_id
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        batch_size = len(queues)
        # 각 요청별로 토큰 캐시와 관련 변수를 초기화
        self.token_caches = [[] for _ in range(batch_size)]
        self.print_lens = [0] * batch_size
        self.next_tokens_are_prompt = [True] * batch_size
        self.finished = [False] * batch_size  # 각 요청별 완료 상태

    def put(self, value):
        # value는 (batch_size,) 또는 (batch_size, 1) 형태의 텐서
        tokens = value.tolist()
        for i, token in enumerate(tokens):
            if self.finished[i]:
                continue  # 이미 완료된 요청은 무시

            if isinstance(token, list):
                token = token[0]

            if token in self.eos_token_id:
                # 종료 토큰이 생성되면 해당 요청을 완료로 표시하고 종료 처리
                self.finished[i] = True
                # 남은 캐시를 디코딩하여 출력
                if len(self.token_caches[i]) > 0:
                    text = self.tokenizer.decode(self.token_caches[i], **self.decode_kwargs)
                    printable_text = text[self.print_lens[i]:]
                    self.token_caches[i] = []
                    self.print_lens[i] = 0
                    if printable_text:
                        asyncio.run_coroutine_threadsafe(self.queues[i].put(printable_text), self.event_loop)
                # 종료 신호 전송
                asyncio.run_coroutine_threadsafe(self.queues[i].put(None), self.event_loop)
                self.next_tokens_are_prompt[i] = True  # 상태 초기화
                continue

            if self.skip_prompt and self.next_tokens_are_prompt[i]:
                self.next_tokens_are_prompt[i] = False
                continue

            # 각 요청별 토큰 캐시에 토큰 추가
            self.token_caches[i].append(token)
            # 현재 토큰 캐시를 디코딩
            text = self.tokenizer.decode(self.token_caches[i], **self.decode_kwargs)

            # 출력 가능한 텍스트 결정
            if text.endswith("\n"):
                # 캐시를 비웁니다.
                printable_text = text[self.print_lens[i]:]
                self.token_caches[i] = []
                self.print_lens[i] = 0
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_lens[i]:]
                self.print_lens[i] += len(printable_text)
            else:
                last_space = text.rfind(" ") + 1
                if last_space > self.print_lens[i]:
                    printable_text = text[self.print_lens[i]:last_space]
                    self.print_lens[i] = last_space
                else:
                    printable_text = ""

            # 출력 가능한 텍스트를 응답 큐에 추가
            if printable_text:
                asyncio.run_coroutine_threadsafe(self.queues[i].put(printable_text), self.event_loop)

    def end(self):
        # 각 요청별로 남은 캐시를 비우고 최종 텍스트를 전송
        for i in range(len(self.queues)):
            if self.finished[i]:
                continue  # 이미 완료된 요청은 무시

            if len(self.token_caches[i]) > 0:
                text = self.tokenizer.decode(self.token_caches[i], **self.decode_kwargs)
                printable_text = text[self.print_lens[i]:]
                self.token_caches[i] = []
                self.print_lens[i] = 0
            else:
                printable_text = ""

            # 최종 텍스트를 응답 큐에 추가
            if printable_text:
                asyncio.run_coroutine_threadsafe(self.queues[i].put(printable_text), self.event_loop)

            # 응답 큐에 None을 넣어 스트리밍 종료를 알림
            asyncio.run_coroutine_threadsafe(self.queues[i].put(None), self.event_loop)
            self.next_tokens_are_prompt[i] = True  # 상태 초기화

    def _is_chinese_char(self, cp):
        """CP가 CJK 문자인지 확인합니다."""
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True
        return False