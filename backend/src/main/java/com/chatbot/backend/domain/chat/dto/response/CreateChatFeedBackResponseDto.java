package com.chatbot.backend.domain.chat.dto.response;

import com.chatbot.backend.domain.chat.dto.ChatDto;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor
@Data
public class CreateChatFeedBackResponseDto {

	private ChatDto chat;
	private String negativeReason;

}
