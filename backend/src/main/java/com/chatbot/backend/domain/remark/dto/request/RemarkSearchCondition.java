package com.chatbot.backend.domain.remark.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "비고 조회 DTO")
public record RemarkSearchCondition(
	@Schema(description = "게시판 ID", example = "1")
	Long boardId,

	@Schema(description = "작성자 이름", example = "김싸피")
	String name,

	@Schema(description = "게시글 내용", example = "게시글 내용")
	String contents
) {
}
