package com.chatbot.backend.global.shared;

import java.time.LocalDateTime;

import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;

import lombok.Getter;

@Getter
public abstract class BaseTimeDocument {

	@CreatedDate
	protected LocalDateTime createdDate;

	@LastModifiedDate
	protected LocalDateTime modifiedDate;
}