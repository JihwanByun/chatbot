package com.chatbot.backend.domain.user.dto.request;

import com.chatbot.backend.global.jwt.Role;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

@Schema(description = "사용자 정보 변경 DTO")
public record UserUpdateRequestDto(
	@Schema(description = "사용자 레벨 (1-3)", example = "1")
	@Min(value = 1, message = "레벨은 1 이상이어야 합니다.")
	@Max(value = 3, message = "레벨은 3 이하여야 합니다.")
	Integer level,

	@Schema(description = "사용자 역할 (USER, USER_WRITE, ADMIN) ", example = "USER_WRITE")
	Role role
) {
}
