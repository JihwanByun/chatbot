package com.chatbot.backend.domain.board.service;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.chatbot.backend.domain.board.exception.BoardNotificationFailException;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

@Slf4j
@Service
@RequiredArgsConstructor
public class BoardNotificationService {
	private final WebClient webClient;

	public void processBoardNotification(MultipartFile file, Integer level) {
		createBoardNotification(file, level)
			.subscribeOn(Schedulers.boundedElastic())
			.subscribe(
				error -> handleSummaryError()
			);
	}

	private Mono<Void> createBoardNotification(MultipartFile file, Integer level) {
		try {
			MultipartBodyBuilder builder = new MultipartBodyBuilder();

			// 파일이 null이 아닌 경우에만 파일 추가
			if (file != null && !file.isEmpty()) {
				byte[] fileContent = file.getBytes();
				builder.part("file", new ByteArrayResource(fileContent) {
					@Override
					public String getFilename() {
						return file.getOriginalFilename();
					}

					@Override
					public long contentLength() {
						return fileContent.length;
					}
				}).filename(file.getOriginalFilename());
			}

			// 레벨이 null이 아닌 경우에만 레벨 추가
			if (level != null) {
				builder.part("level", String.valueOf(level));
			}

			return webClient
				.post()
				.uri("/update")
				.contentType(MediaType.MULTIPART_FORM_DATA)
				.body(BodyInserters.fromMultipartData(builder.build()))
				.retrieve()
				.bodyToMono(String.class)
				.doOnNext(response -> log.info("Response from FastAPI: {}", response))
				.then()
				.doOnError(e -> log.error("Error sending request to FastAPI: ", e))
				.doOnSuccess(v -> log.info("Request sent to FastAPI successfully"));
		} catch (Exception e) {
			throw new BoardNotificationFailException(e);
		}
	}

	private void handleSummaryError() {
		throw new BoardNotificationFailException();
	}
}
