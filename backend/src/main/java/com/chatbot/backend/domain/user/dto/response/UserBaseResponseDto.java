package com.chatbot.backend.domain.user.dto.response;

import com.chatbot.backend.domain.user.entity.User;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Builder
@Schema(description = "사용자 기본 정보 응답 Dto")
public record UserBaseResponseDto(
	Long userId,
	String email,
	String name,
	String employeeNum,
	String department,
	Integer level,
	String role
) {
	public static UserBaseResponseDto of(User user) {
		return UserBaseResponseDto.builder()
			.userId(user.getId())
			.email(user.getEmail())
			.name(user.getName())
			.employeeNum(user.getEmployeeNum())
			.department(user.getDepartment())
			.level(user.getLevel())
			.role(user.getRole().getKey())
			.build();
	}
}
