package com.chatbot.backend.domain.user.dto.request;

import com.chatbot.backend.global.jwt.Role;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "사용자 검색 조건 DTO")
public record UserSearchCondition(
	@Schema(description = "사용자 이메일", example = "ssafy@com")
	String email,

	@Schema(description = "사용자 이름", example = "김싸피")
	String name,

	@Schema(description = "사용자 사번", example = "12345678")
	String employeeNum,

	@Schema(description = "사용자 부서", example = "인사과")
	String department,

	@Schema(description = "사용자 정보 확인 레벨 (1-3)", example = "1")
	Integer level,

	@Schema(description = "사용자 역할 (USER, USER_WRITE, ADMIN)", example = "USER")
	Role role
) {
}
