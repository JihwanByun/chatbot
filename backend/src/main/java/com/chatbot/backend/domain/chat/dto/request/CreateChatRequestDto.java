package com.chatbot.backend.domain.chat.dto.request;

import com.chatbot.backend.domain.chat.entity.source.Memory;

import lombok.Getter;

@Getter
public class CreateChatRequestDto {

	private Long chatRoomId;
	private Memory memory;

	CreateChatRequestDto(Long chatRoomId, Memory memory) {
		this.chatRoomId = chatRoomId;
		this.memory = memory;
	}
}
