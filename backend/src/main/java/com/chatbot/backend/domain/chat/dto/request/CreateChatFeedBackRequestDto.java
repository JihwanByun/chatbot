package com.chatbot.backend.domain.chat.dto.request;

import lombok.Data;

@Data
public class CreateChatFeedBackRequestDto {

	private Boolean isPositive;
	private String negativeReason;

}
