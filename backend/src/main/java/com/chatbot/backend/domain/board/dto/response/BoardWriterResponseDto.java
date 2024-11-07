package com.chatbot.backend.domain.board.dto.response;

import com.chatbot.backend.domain.user.entity.User;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Schema(description = "게시글 작성자 정보 DTO")
@Builder
public record BoardWriterResponseDto(
	@Schema(description = "작성자 이메일", example = "user@example.com")
	String email,

	@Schema(description = "작성자 이름", example = "이싸피")
	String name,

	@Schema(description = "프로필 이미지 URL", example = "https://example.com/profiles/1")
	String profileUrl
) {
	public static BoardWriterResponseDto of(User user) {
		return BoardWriterResponseDto.builder()
			.email(user.getEmail())
			.name(user.getName())
			.profileUrl(user.getProfileUrl())
			.build();
	}
}
