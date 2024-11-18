package com.chatbot.backend.domain.file.dto;

import java.util.List;

import com.chatbot.backend.domain.regulation.entity.Regulation;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Builder
@Schema(description = "PDF 정보 응답 DTO")
public record PdfRequestDto(
	@Schema(description = "추출된 PDF 제목", example = "PDF 제목입니다.")
	String title,

	@Schema(description = "추출된 PDF 내용", example = "PDF 내부 내용입니다.ㅌ")
	String text,

	@Schema(description = "게시글 레벨", example = "1")
	Integer level,

	@Schema(description = "추출된 PDF 제목", example = "PDF 제목입니다.")
	List<Regulation> regulationList
) {
	public static PdfRequestDto of(String text, String title, Integer level, List<Regulation> regulationList) {
		return PdfRequestDto.builder()
			.text(text == null ? "" : text)
			.title(title == null ? "" : title)
			.level(level)
			.regulationList(regulationList)
			.build();
	}
}