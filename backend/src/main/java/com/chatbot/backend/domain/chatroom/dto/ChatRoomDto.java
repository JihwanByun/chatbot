package com.chatbot.backend.domain.chatroom.dto;

import java.time.LocalDateTime;

import lombok.Data;

@Data
public class ChatRoomDto {

	private Long chatRoomId;
	private String title;
	private LocalDateTime createdAt;

	public ChatRoomDto(Long chatRoomId, String title, LocalDateTime createdAt) {
		this.chatRoomId = chatRoomId;
		this.title = title;
		this.createdAt = createdAt;
	}
}
