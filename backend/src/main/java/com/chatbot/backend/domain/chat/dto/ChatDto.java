package com.chatbot.backend.domain.chat.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ChatDto {

	private String chatId;
	private String question;
	private String answer;

}
