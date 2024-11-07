package com.chatbot.backend.domain.chat.dto.request;

import com.chatbot.backend.domain.chat.entity.source.Memory;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class CreateChatRequestDto {

	private Long chatRoomId;
	private Memory memory;

}
