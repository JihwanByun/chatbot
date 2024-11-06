package com.chatbot.backend.domain.chat.dto;

import com.chatbot.backend.domain.chat.entity.source.Memory;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ChatDto {

	private String chatId;
	private Memory memory;
	private Boolean isPositive;
}
