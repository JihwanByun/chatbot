package com.chatbot.backend.domain.feedback.service;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.chatbot.backend.domain.feedback.dto.FeedbackResponseDto;
import com.chatbot.backend.domain.feedback.repository.FeedbackRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class FeedbackServiceImpl implements FeedbackService {
	private final FeedbackRepository feedbackRepository;

	/**
	 * 피드백 정보 목록 조회
	 *
	 * @param userId
	 * @return
	 */
	@Override
	public List<FeedbackResponseDto> getFeedbackList(Long userId) {
		return feedbackRepository.findAllByIsDeletedFalse().stream().map(FeedbackResponseDto::of).toList();
	}
}
