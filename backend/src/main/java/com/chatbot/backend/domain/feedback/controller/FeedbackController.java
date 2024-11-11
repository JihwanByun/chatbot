package com.chatbot.backend.domain.feedback.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.chatbot.backend.domain.feedback.dto.FeedbackResponseDto;
import com.chatbot.backend.domain.feedback.service.FeedbackService;
import com.chatbot.backend.global.security.CustomUserDetails;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Tag(name = "Feedback", description = "피드백 관련 API")
@RestController
@RequiredArgsConstructor
@Slf4j
@RequestMapping("/api/feedbacks")
public class FeedbackController {
	private final FeedbackService feedbackService;

	@Operation(
		summary = "피드백 목록 조회",
		description = "피드백 목록을 조회합니다."
	)
	@GetMapping
	public ResponseEntity<List<FeedbackResponseDto>> getFeedbackList(
		@AuthenticationPrincipal CustomUserDetails userDetails
	) {
		return ResponseEntity.status(HttpStatus.OK).body(feedbackService.getFeedbackList(userDetails.getId()));
	}
}
