package com.chatbot.backend.domain.board.dto.request;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.regulation.dto.request.RegulationRequestDto;
import com.chatbot.backend.domain.user.entity.User;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

// 게시글 생성 요청을 위한 DTO
// record로 사용하여 불변 객체로 구현
@Schema(description = "게시글 생성 DTO")
public record BoardCreateRequestDto(
	@Schema(description = "게시글 제목", example = "게시글 제목입니다", required = true)
	@NotBlank(message = "제목은 필수입니다.")
	@Size(max = 200, message = "제목은 200자를 초과할 수 없습니다.")
	String title,

	@Schema(description = "게시글 내용", example = "게시글 내용입니다", required = false, nullable = true)
	String contents,

	@Schema(description = "게시글 카테고리", example = "카테고리 1", required = true)
	@NotBlank(message = "카테고리는 필수입니다.")
	String category,

	@Schema(description = "게시글 레벨 (1-3)", example = "1", required = true)
	@NotNull(message = "레벨은 필수입니다.")
	@Min(value = 1, message = "레벨은 1 이상이어야 합니다.")
	@Max(value = 3, message = "레벨은 3 이하여야 합니다.")
	Integer level,

	@Schema(description = "규정 정보", required = false, nullable = true)
	RegulationRequestDto regulationRequest,

	@Schema(description = "파일 첨부 여부", example = "false", required = false)
	Boolean hasFile
) {
	// Board 엔티티로 변환하는 메소드
	public Board toEntity(User user, Category category, String fileUrl) {
		return Board.builder()
			.title(title)
			.contents(contents)
			.level(level)

			.user(user)

			.category(category)

			.fileUrl(fileUrl)

			.hasFile(hasFile != null ? hasFile : null)
			.build();
	}

	public Board toEntity(User user, Category category) {
		return toEntity(user, category, null);
	}
}