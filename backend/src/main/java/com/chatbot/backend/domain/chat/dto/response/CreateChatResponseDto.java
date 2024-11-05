package com.chatbot.backend.domain.chat.dto.response;

import com.chatbot.backend.domain.chat.entity.source.Memory;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
@AllArgsConstructor
public class CreateChatResponseDto {

	private String chatId;
	private Memory memory;
}
