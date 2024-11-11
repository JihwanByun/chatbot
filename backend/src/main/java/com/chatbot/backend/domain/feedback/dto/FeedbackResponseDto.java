package com.chatbot.backend.domain.feedback.dto;

import com.chatbot.backend.domain.feedback.entity.Feedback;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Builder
@Schema(description = "피드백 응답 DTO")
public record FeedbackResponseDto(
	@Schema(description = "피드백 내용", example = "피드백 내용입니다.")
	String name
) {
	// 피드백 정보로 응답 DTO 생성
	public static FeedbackResponseDto of(Feedback feedback) {
		return FeedbackResponseDto.builder()
			.name(feedback.getName())
			.build();
	}
}