package com.chatbot.backend.domain.chat.dto.request;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Data
public class CreateChatFeedBackRequestDto {

	private boolean isPositive;
	private String negativeReason;

}
