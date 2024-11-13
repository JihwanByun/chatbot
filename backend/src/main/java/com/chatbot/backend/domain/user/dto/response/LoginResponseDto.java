package com.chatbot.backend.domain.user.dto.response;

import com.chatbot.backend.domain.user.entity.User;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Builder
@Schema(description = "로그인 시 응답을 위한 DTO")
public record LoginResponseDto(
	String token,
	String name,
	String employeeNum,
	String department,
	Integer level,
	String role
) {
	public static LoginResponseDto of(String token, User user) {
		if (token == null || user == null) {
			return null;
		}

		return LoginResponseDto.builder()
			.token(token)
			.name(user.getName())
			.employeeNum(user.getEmployeeNum())
			.department(user.getDepartment())
			.level(user.getLevel())
			.role(String.valueOf(user.getRole()))
			.build();
	}
}
