package com.chatbot.backend.domain.feedback.service;

import java.util.List;

import com.chatbot.backend.domain.feedback.dto.FeedbackResponseDto;

public interface FeedbackService {
	List<FeedbackResponseDto> getFeedbackList(Long userId);
}
