package com.chatbot.backend.domain.board.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "게시글 조회 DTO")
public record BoardSearchCondition(
	@Schema(description = "게시글 레벨", example = "1")
	Integer level,

	@Schema(description = "작성자 이름", example = "김싸피")
	String name,

	@Schema(description = "게시글 제목", example = "게시글 제목")
	String title
) {
}
