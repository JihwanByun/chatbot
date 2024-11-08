package com.chatbot.backend.domain.regulation.dto.request;

import java.util.List;

import com.chatbot.backend.domain.regulation.entity.RegulationItem;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "규정 항목 요청을 위한 DTO")
public record RegulationItemRequestDto(
	@Schema(description = "규정 항목 번호", example = "1")
	Integer number,

	@Schema(description = "규정 내용", example = "이것은 규정입니다.")
	String content,

	@Schema(description = "하위 규정 목록")
	List<RegulationItemRequestDto> itemList
) {
	// 규정 항목을 Document로 변환
	public RegulationItem toDocument() {
		return RegulationItem.builder()
			.number(number)
			.content(content)
			.itemList(itemList != null ?
				itemList.stream()
					.map(RegulationItemRequestDto::toDocument)
					.toList() :
				null)
			.build();
	}
}