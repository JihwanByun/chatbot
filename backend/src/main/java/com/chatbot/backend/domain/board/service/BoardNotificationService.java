package com.chatbot.backend.domain.board.service;

import java.util.List;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.chatbot.backend.domain.board.exception.BoardNotificationFailException;
import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.file.dto.PdfRequestDto;
import com.chatbot.backend.domain.regulation.entity.Regulation;
import com.chatbot.backend.domain.regulation.repository.RegulationRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

/**
 * 게시판 알림 처리를 담당하는 서비스
 * PDF 파일에서 텍스트를 추출한 후, FAST API에 데이터를 전송하여 알람 처리를 수행
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class BoardNotificationService {
	private final WebClient webClient;
	private final RegulationRepository regulationRepository;

	/**
	 * 게시판 알림을 처리
	 * PDF 파일과 레벨을 받아서 알림 생성 요청을 수행
	 * @param file    알림 생성에 필요한 PDF 파일
	 * @param level 게시글 접근 수준을 나타내는 레벨
	 */
	public void processBoardNotification(MultipartFile file, String title, Integer level) {
		createBoardNotification(file, title, level)    // 알림 생성 요청을 실행
			.subscribeOn(Schedulers.boundedElastic())
			.subscribe(
				success -> log.info("Notification processed successfully"),
				error -> handleSummaryError()    // 오류 시 예외 처리
			);
	}

	/**
	 * 게시글 알림 생성 요청을 수행
	 * PDF 에서 텍스트를 추출하고 DTO를 생성
	 * @param file
	 * @return
	 */
	private String extractTextFromPdf(MultipartFile file) {
		try (PDDocument document = PDDocument.load(file.getInputStream())) {
			PDFTextStripper stripper = new PDFTextStripper();
			return stripper.getText(document);
		} catch (Exception e) {
			log.error("Error extracting text from PDF: ", e);
			throw new RuntimeException();
		}
	}

	/**
	 * 게시판 알림 생성 요청을 수행
	 * PDF에서 텍스트를 추출하고 DTO를 생성하여 FastAPI로 전송
	 * @param file 텍스트를 추출할 PDF 파일
	 * @param level 게시글의 접근 수준
	 * @return FastAPI 요청에 대한 Mono 객체
	 */
	private Mono<Void> createBoardNotification(MultipartFile file, String title, Integer level) {
		try {
			String pdfText = null;

			if (file != null) {
				// PDF에서 텍스트 추출
				pdfText = extractTextFromPdf(file);
				log.info(pdfText);
			}

			List<Regulation> regulationList = regulationRepository.findAll();

			// Request DTO 생성
			PdfRequestDto requestDto = PdfRequestDto.of(pdfText, title, level, regulationList);
			log.info("RequestDto 정보 : (Text)" + requestDto.text());
			log.info("RequestDto 정보 : (Title)" + requestDto.title());
			log.info("RequestDto 정보 : (Level)" + requestDto.level());

			// FastAPI 에 POST 요청을 보내 알림 정보를 전송
			return webClient
				.post()
				.uri("/update")
				.contentType(MediaType.APPLICATION_JSON)    // JSON 형식의 콘텐츠 타입
				.body(BodyInserters.fromValue(requestDto))    // 요청 Body에 DTO 삽입
				.retrieve()
				.bodyToMono(Void.class)
				.doOnNext(response -> log.info("Response from FastAPI: {}", response))
				.then()
				.doOnError(e -> log.error("Error sending request to FastAPI: ", e))
				.doOnSuccess(v -> log.info("Request sent to FastAPI successfully"));
		} catch (Exception e) {
			throw new BoardNotificationFailException(e);
		}
	}

	/**
	 * 알림 생성 중 오류가 발생했을 때 예외를 발생시키는 메소드
	 */
	private void handleSummaryError() {
		throw new BoardNotificationFailException();
	}
}