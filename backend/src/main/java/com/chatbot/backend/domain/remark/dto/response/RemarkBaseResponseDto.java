package com.chatbot.backend.domain.remark.dto.response;

import java.time.LocalDateTime;

import com.chatbot.backend.domain.remark.entity.Remark;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

// 비고 수정 요청을 위한 DTO
// record로 사용하여 불변 객체로 구현
@Schema(description = "비고 수정 DTO")
@Builder
public record RemarkBaseResponseDto(
	@Schema(description = "비고 ID", example = "1")
	Long remarkId,

	@Schema(description = "비고 내용", example = "비고의 내용입니다.")
	String contents,

	@Schema(description = "작성자 이름", example = "이싸피")
	String name,

	@Schema(description = "비고 생성 시간")
	LocalDateTime createdAt,

	@Schema(description = "게시판 ID", example = "1")
	Long boardId
) {
	// 응답 DTO 생성
	public static RemarkBaseResponseDto of(Remark remark) {
		return RemarkBaseResponseDto.builder()
			.remarkId(remark.getId())
			.contents(remark.getContents())
			.name(remark.getUser().getName())
			.createdAt(remark.getCreatedAt())
			.boardId(remark.getBoard().getId())
			.build();
	}
}
