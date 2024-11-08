package com.chatbot.backend.domain.regulation.dto.response;

import java.util.List;

import com.chatbot.backend.domain.regulation.entity.Regulation;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;

@Builder
@Schema(description = "규정 항목 번호")
public record RegulationResponseDto(
	@Schema(description = "규정 ID")
	String id,

	@Schema(description = "관련 게시글 ID")
	Long boardId,

	@Schema(description = "규정 항목 목록")
	List<RegulationItemResponseDto> itemList
) {
	// Regulation 객체를 RegulationResponseDto로 변환
	public static RegulationResponseDto of(Regulation regulation) {
		if (regulation == null)
			return null;
		return RegulationResponseDto.builder()
			.id(regulation.getId())
			.boardId(regulation.getBoardId())
			.itemList(regulation.getItemList().stream()
				.map(RegulationItemResponseDto::of)
				.toList())
			.build();
	}
}
