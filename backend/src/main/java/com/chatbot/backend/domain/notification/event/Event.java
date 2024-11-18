package com.chatbot.backend.domain.notification.event;

import lombok.Getter;

@Getter
public enum Event {
	UPDATE_VECTOR_DB("update.vector.db"),
	SUMMARY_FILE("summary.file");

	private String name;

	Event(String name){
		this.name = name;
	}
}
