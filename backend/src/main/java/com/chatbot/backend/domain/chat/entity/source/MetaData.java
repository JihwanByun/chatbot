package com.chatbot.backend.domain.chat.entity.source;

import lombok.Getter;

@Getter
public class MetaData {
	private String fileName;
	private int totalPages;
	private int page;
	private String category;
	private int id;
	private int level;
}
