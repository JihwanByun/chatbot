package com.chatbot.backend.domain.regulation.dto.response;

import java.util.List;

import com.chatbot.backend.domain.regulation.entity.RegulationItem;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Builder
@Schema(description = "규정 항목 응답을 위한 DTO")
public record RegulationItemResponseDto(
	@Schema(description = "규정 항목 번호")
	Integer number,

	@Schema(description = "규정 내용")
	String content,

	@Schema(description = "하위 규정 목록")
	List<RegulationItemResponseDto> itemList
) {
	// RegulationItem 객체를 RegulationItemResponseDto로 변환
	public static RegulationItemResponseDto of(RegulationItem item) {
		if (item == null) {
			return null;
		}

		return RegulationItemResponseDto.builder()
			.number(item.getNumber())
			.content(item.getContent())
			.itemList(item.getItemList() != null ?
				item.getItemList().stream()
					.map(RegulationItemResponseDto::of)
					.toList() :
				null)
			.build();
	}
}