package com.chatbot.backend.domain.chat.entity;

import java.time.LocalDateTime;

import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import com.chatbot.backend.domain.chat.entity.source.Memory;

import jakarta.persistence.Column;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Document(collection = "chats")
@Getter
public class Chat {

	@Id
	private ObjectId chatId;
	private LocalDateTime createdAt;
	@Setter
	private Long chatRoomId;

	private Memory memory;

	@Setter
	@Column(nullable = true)
	private Boolean isPositive;

	@Builder
	public Chat(Memory memory, Long chatRoomId) {
		this.memory = memory;
		this.chatRoomId = chatRoomId;
		this.createdAt = LocalDateTime.now();
	}

}
