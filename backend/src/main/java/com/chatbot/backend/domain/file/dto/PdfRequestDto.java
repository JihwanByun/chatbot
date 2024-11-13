package com.chatbot.backend.domain.file.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;

@Builder
@Schema(description = "PDF 정보 응답 DTO")
public record PdfRequestDto(
	@Schema(description = "추출된 PDF 내용", example = "PDF 내부 내용입니다. ")
	String text,

	@Schema(description = "게시글 레벨", example = "1")
	Integer level
) {
	public static PdfRequestDto of(String text, Integer level) {
		return PdfRequestDto.builder()
			.text(text == null ? "" : text)
			.level(level)
			.build();
	}
}