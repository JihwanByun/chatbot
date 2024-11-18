package com.chatbot.backend.domain.file.service;

import java.util.HashMap;
import java.util.Map;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.file.exception.FileSummaryFailException;
import com.chatbot.backend.domain.notification.event.Event;
import com.chatbot.backend.domain.notification.service.NotificationService;
import com.mongodb.internal.VisibleForTesting;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

@Slf4j
@Service
@RequiredArgsConstructor
public class FileSummaryService {
	private final WebClient webClient;
	private final BoardRepository boardRepository;
	private final NotificationService notificationService;

	public void processFileSummaryAsync(MultipartFile file, Long boardId) {
		String fileName = file.getOriginalFilename();
		getSummary(file)
			.subscribeOn(Schedulers.boundedElastic())
			.subscribe(
				summary -> {
					Map<String, Object> data = new HashMap<>();
					data.put("fileName", fileName);
					notificationService.notifyAll(Event.SUMMARY_FILE, "규정 파일이 요약되었습니다.", data);
					updateBoardSummary(boardId, summary.substring(1, summary.length() - 1));
				},
				error -> handleSummaryError(boardId, error)
			);
	}

	private Mono<String> getSummary(MultipartFile file) {
		try {
			byte[] fileContent = file.getBytes();
			MultipartBodyBuilder builder = new MultipartBodyBuilder();
			builder.part(
					"file",
					new ByteArrayResource(fileContent) {
						@Override
						public String getFilename() {
							return "file.pdf";
						}

						@Override
						public long contentLength() {
							return fileContent.length;
						}
					}
				)
				.filename("file.pdf");

			return webClient
				.post()
				.uri("/summarize")
				.contentType(MediaType.MULTIPART_FORM_DATA)
				.body(BodyInserters.fromMultipartData(builder.build()))
				.retrieve()
				.bodyToMono(String.class);
		} catch (Exception e) {
			throw new FileSummaryFailException(e);
		}

	}

	private void updateBoardSummary(Long boardId, String summary) {
		try {
			Board board = boardRepository.findByIdOrElseThrow(boardId);
			board.updateContents(summary);
			boardRepository.save(board);
			log.info("게시글 ID: {} 요약 업데이트 완료", boardId);
		} catch (Exception e) {
			throw new FileSummaryFailException(e);
		}
	}

	private void handleSummaryError(Long boardId, Throwable error) {
		throw new FileSummaryFailException();
	}
}