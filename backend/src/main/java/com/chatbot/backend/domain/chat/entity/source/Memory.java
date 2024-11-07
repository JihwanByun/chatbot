package com.chatbot.backend.domain.chat.entity.source;

import java.util.List;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class Memory {

	private String question;
	private List<Doc> docs;
	private String answer;

	public Memory(Memory memory) {
		this.question = memory.question;
		this.docs = memory.docs;
		this.answer = memory.answer;
	}
}
