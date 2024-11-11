package com.chatbot.backend.domain.remark.dto.request;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.remark.entity.Remark;
import com.chatbot.backend.domain.user.entity.User;

import io.swagger.v3.oas.annotations.media.Schema;

// 비고 생성 요청을 위한 DTO
// record로 사용하여 불변 객체로 구현
@Schema(description = "비고 생성 DTO")
public record RemarkCreateRequestDto(
	@Schema(description = "게시판 ID", example = "1")
	Long boardId,

	@Schema(description = "비고 내용", example = "비고의 내용입니다.")
	String contents
) {
	// Remark 엔티티로 변환하는 메소드
	public Remark toDocument(User user, Board board) {
		return Remark.builder()
			.contents(contents)
			.user(user)
			.board(board)
			.build();
	}
}
