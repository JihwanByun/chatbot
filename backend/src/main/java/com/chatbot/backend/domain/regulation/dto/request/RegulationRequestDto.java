package com.chatbot.backend.domain.regulation.dto.request;

import java.util.List;

import com.chatbot.backend.domain.regulation.entity.Regulation;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "규정 요청을 위한 DTO")
public record RegulationRequestDto(
	@Schema(description = "규정 목록")
	List<RegulationItemRequestDto> itemList
) {
	// 규정을 Document로 변환
	public Regulation toDocument(Long boardId, String title, Integer level) {
		return Regulation.builder()
			.boardId(boardId)
			.title(title)
			.level(level)
			.itemList(itemList.stream().map(RegulationItemRequestDto::toDocument).toList())
			.build();
	}
}

