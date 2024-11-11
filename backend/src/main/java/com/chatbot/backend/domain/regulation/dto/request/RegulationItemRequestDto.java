package com.chatbot.backend.domain.regulation.dto.request;

import java.util.List;

import com.chatbot.backend.domain.regulation.entity.RegulationItem;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Column;

@Schema(description = "규정 항목 요청을 위한 DTO")
public record RegulationItemRequestDto(
	@Column(nullable = true)
	@Schema(description = "규정 제목", example = "이것은 규정 제목입니다.")
	String title,

	@Column(nullable = true)
	@Schema(description = "규정 내용", example = "이것은 규정입니다.")
	String content,

	@Column(nullable = true)
	@Schema(description = "하위 규정 목록")
	List<RegulationItemRequestDto> itemList
) {
	// 규정 항목을 Document로 변환
	public RegulationItem toDocument() {
		return RegulationItem.builder()
			.title(title)
			.content(content)
			.itemList(itemList != null ?
				itemList.stream()
					.map(RegulationItemRequestDto::toDocument)
					.toList() :
				null)
			.build();
	}
}