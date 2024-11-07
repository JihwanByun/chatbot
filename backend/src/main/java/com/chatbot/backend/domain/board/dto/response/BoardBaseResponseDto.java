package com.chatbot.backend.domain.board.dto.response;

import java.time.LocalDateTime;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.entity.BoardLike;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

// 게시글 기본 정보 응답을 위한 DTO
// 게시글, 작성자, 좋아요 정보를 포함
@Builder
@Schema(description = "게시글 기본 정보 응답 DTO")
public record BoardBaseResponseDto(
	// Board
	@Schema(description = "게시글 ID", example = "1")
	Long boardId,

	@Schema(description = "게시글 제목", example = "게시글 제목입니다")
	String title,

	@Schema(description = "게시글 내용", example = "게시글 내용입니다")
	String contents,

	@Schema(description = "게시글 생성 일시", example = "2024-10-31T10:30:00")
	LocalDateTime createdAt,

	// User
	@Schema(description = "작성자 이름", example = "이싸피")
	String name,

	// BoardLike
	@Schema(description = "좋아요 여부", example = "false")
	Boolean isFavorite
) {
	// 게시글, 사용자, 좋아요 정보를 조합하여 응답 DTO 생성
	public static BoardBaseResponseDto of(Board board, BoardLike boardLike) {
		return BoardBaseResponseDto.builder()
			.boardId(board.getId())
			.title(board.getTitle())
			.contents(board.getContents())
			.createdAt(board.getCreatedAt())

			.name(board.getUser().getName())

			.isFavorite(boardLike == null ? false : true)
			.build();
	}

	public static BoardBaseResponseDto of(Board board) {
		return BoardBaseResponseDto.of(board, null);
	}
}