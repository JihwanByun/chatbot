package com.chatbot.backend.domain.chat.entity;

import java.time.LocalDateTime;

import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import jakarta.persistence.Column;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Document(collection = "chats")
@Getter
public class Chat {

	@Id
	private ObjectId chatId;

	private String question;
	private String answer;

	private LocalDateTime createdAt;
	@Setter
	private Long chatRoomId;

	@Setter
	@Column(nullable = true)
	private Boolean isPositive;

	@Builder
	public Chat(String question, String answer, Long chatRoomId) {
		this.question = question;
		this.answer = answer;
		this.chatRoomId = chatRoomId;
		this.createdAt = LocalDateTime.now();
	}

}
