package com.chatbot.backend.domain.admin.dto.response;

import java.time.LocalDateTime;

import com.chatbot.backend.domain.chat.entity.ChatFeedBack;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

@Getter
@Builder
@ToString
public class FeedbackResponseDto {
	private String chatId;
	private String question;
	private String answer;
	private Boolean isPositive;
	private String negativeReason;
	private LocalDateTime createdAt;

	public static FeedbackResponseDto of(ChatFeedBack chatFeedBack){
		return
			FeedbackResponseDto.builder()
				.chatId(String.valueOf(chatFeedBack.getChat().getChatId()))
				.question(chatFeedBack.getChat().getQuestion())
				.answer(chatFeedBack.getChat().getAnswer())
				.isPositive(chatFeedBack.getChat().getIsPositive())
				.negativeReason(chatFeedBack.getNegativeReason())
				.createdAt(LocalDateTime.now())
				.build();
	}

}
