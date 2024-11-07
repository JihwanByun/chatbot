package com.chatbot.backend.domain.board.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

// 게시글 수정 요청을 위한 DTO
// record로 사용하여 불변 객체로 구현
@Schema(description = "게시글 수정 DTO")
public record BoardUpdateRequestDto(
	@Schema(description = "게시글 제목", example = "수정할 게시글 제목입니다", required = true)
	@NotBlank(message = "제목은 필수입니다.")
	@Size(max = 200, message = "제목은 200자를 초과할 수 없습니다.")
	String title,

	@Schema(description = "게시글 내용", example = "수정할 게시글 내용입니다", required = true)
	@NotBlank(message = "내용은 필수입니다.")
	String contents,

	@Schema(description = "게시글 카테고리", example = "수정할 카테고리 1", required = true)
	@NotBlank(message = "카테고리는 필수입니다.")
	String category,

	@Schema(description = "게시글 레벨 (1-3)", example = "1", required = true)
	@NotNull(message = "레벨은 필수입니다.")
	@Min(value = 1, message = "레벨은 1 이상이어야 합니다.")
	@Max(value = 3, message = "레벨은 3 이하여야 합니다.")
	Integer level
) {
}