package com.chatbot.backend.domain.category.dto.response;

import com.chatbot.backend.domain.category.entity.Category;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

@AllArgsConstructor
@Getter
@ToString
@Builder
public class CategoryItemDto {
	private Long categoryId;
	private String name;

	// Category 객체를 CategoryItemDto로 변환
	public static CategoryItemDto of(Category category) {
		return CategoryItemDto.builder()
			.categoryId(category.getId())
			.name(category.getName())
			.build();
	}
}
