package com.chatbot.backend.domain.file.service;

import java.io.IOException;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.repository.BoardRepository;

import jakarta.persistence.EntityNotFoundException;
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

	public void processFileSummaryAsync(MultipartFile file, Long boardId) throws IOException {
		getSummary(file.getBytes())
			.subscribeOn(Schedulers.boundedElastic())
			.subscribe(
				summary -> updateBoardSummary(boardId, summary),
				error -> handleSummaryError(boardId, error)
			);
	}

	private Mono<String> getSummary(byte[] fileContent) {
		try {
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
			return Mono.error(new RuntimeException("파일 처리 중 오류가 발생했습니다.", e));
		}

	}

	private void updateBoardSummary(Long boardId, String summary) {
		try {
			Board board = boardRepository.findById(boardId)
				.orElseThrow(() -> new EntityNotFoundException("게시글을 찾을 수 없습니다."));
			board.updateContents(summary);
			boardRepository.save(board);
			log.info("게시글 ID: {} 요약 업데이트 완료", boardId);
		} catch (Exception e) {
			log.error("게시글 요약 업데이트 중 오류 발생. 게시글 ID: {}", boardId, e);
		}
	}

	private void handleSummaryError(Long boardId, Throwable error) {
		log.error("게시글 ID: {} 요약 처리 중 오류 발생", boardId, error);
	}
}