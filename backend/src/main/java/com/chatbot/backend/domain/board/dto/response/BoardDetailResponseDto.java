package com.chatbot.backend.domain.board.dto.response;

import java.time.LocalDateTime;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.entity.BoardLike;
import com.chatbot.backend.domain.regulation.dto.response.RegulationResponseDto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

// 게시글 상세 정보 응답을 위한 DTO
// 게시글, 작성자, 좋아요 정보를 포함
@Builder
@Schema(description = "게시글 상세 정보 응답 DTO")
public record BoardDetailResponseDto(
	// Board
	@Schema(description = "게시글 ID", example = "1")
	Long boardId,

	@Schema(description = "게시글 제목", example = "게시글 제목입니다")
	String title,

	@Schema(description = "게시글 내용", example = "게시글 내용입니다")
	String contents,

	@Schema(description = "게시글 레벨", example = "1")
	Integer level,

	@Schema(description = "게시글 카테고리", example = "카테고리 1")
	String category,

	@Schema(description = "첨부 파일 URL", example = "https://example.com/files/1")
	String fileUrl,

	@Schema(description = "게시글 생성 일시", example = "2024-10-31T10:30:00")
	LocalDateTime createdAt,

	// User
	@Schema(description = "작성자 정보")
	BoardWriterResponseDto writer,

	// BoardLike
	@Schema(description = "좋아요 여부", example = "false")
	Boolean isFavorite,

	// Regulation
	@Schema(description = "규정 정보")
	RegulationResponseDto regulationResponseDto,

	@Schema(description = "파일 첨부 여부", example = "false")
	Boolean hasFile
) {
	// 게시글, 사용자, 좋아요 정보를 조합하여 응답 DTO 생성
	public static BoardDetailResponseDto of(Board board, BoardLike boardLike,
		RegulationResponseDto regulationResponseDto) {
		return BoardDetailResponseDto.builder()
			.boardId(board.getId())
			.title(board.getTitle())
			.contents(board.getContents())
			.level(board.getLevel())
			.category(board.getCategory().getName())
			.fileUrl(board.getFileUrl())
			.createdAt(board.getCreatedAt())

			.writer(BoardWriterResponseDto.of(board.getUser()))

			.isFavorite(boardLike == null ? false : true)

			.hasFile(board.getHasFile())
			.regulationResponseDto(regulationResponseDto)
			.build();
	}

	// Board와 규정 정보를 이용해 BoardDetailResponseDto 생성
	public static BoardDetailResponseDto of(Board board, RegulationResponseDto regulationResponseDto) {
		return BoardDetailResponseDto.of(board, null, regulationResponseDto);
	}
}