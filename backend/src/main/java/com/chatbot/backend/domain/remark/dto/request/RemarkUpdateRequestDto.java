package com.chatbot.backend.domain.remark.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;

// 비고 수정 요청을 위한 DTO
// record로 사용하여 불변 객체로 구현
@Schema(description = "비고 수정 DTO")
public record RemarkUpdateRequestDto(
	@Schema(description = "비고 내용", example = "비고의 내용입니다.")
	String contents
) {
}
