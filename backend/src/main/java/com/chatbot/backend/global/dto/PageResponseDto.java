package com.chatbot.backend.global.dto;

import java.util.List;

import org.springframework.data.domain.Page;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class PageResponseDto<T> {
	private List<T> contents;
	private Integer page;
	private Integer size;
	private Integer totalPages;
	private Long totalElements;

	public static <T> PageResponseDto<T> of(Page<T> pageInfo) {
		return PageResponseDto.<T>builder()
			.contents(pageInfo.getContent())
			.page(pageInfo.getPageable().getPageNumber())
			.size(pageInfo.getPageable().getPageSize())
			.totalPages(pageInfo.getTotalPages())
			.totalElements(pageInfo.getTotalElements())
			.build();

	}
}
