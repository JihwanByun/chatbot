package com.chatbot.backend.domain.admin.dto.response;

import lombok.Builder;
import lombok.Getter;
import java.util.List;

import org.springframework.data.domain.Page;

@Getter
@Builder
public class PageResponseDto {
	private List<FeedbackResponseDto> contents;
	private Integer page;
	private Integer size;
	private Integer totalPages;
	private Long totalElements;

	public static PageResponseDto of(Page<FeedbackResponseDto> pageInfo){
		return PageResponseDto.builder()
			.contents(pageInfo.getContent())
			.page(pageInfo.getPageable().getPageNumber())
			.size(pageInfo.getPageable().getPageSize())
			.totalPages(pageInfo.getTotalPages())
			.totalElements(pageInfo.getTotalElements())
			.build();

	}
}
