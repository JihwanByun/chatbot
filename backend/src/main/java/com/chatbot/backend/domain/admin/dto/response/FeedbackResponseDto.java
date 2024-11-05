package com.chatbot.backend.domain.admin.dto.response;

import lombok.Builder;
import lombok.Getter;
import java.util.List;

@Getter
@Builder
public class FeedbackResponseDto {
	private List<ContentDto> contents;
	private Integer page;
	private Integer size;
	private Integer totalPages;
	private Long totalElements;
}
