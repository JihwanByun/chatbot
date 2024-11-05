package com.chatbot.backend.domain.chat.entity.source;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class Doc {

	private MetaData metaData;
	private String pageContent;
}
