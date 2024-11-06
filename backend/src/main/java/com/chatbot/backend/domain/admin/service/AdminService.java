package com.chatbot.backend.domain.admin.service;



import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.admin.dto.response.PageResponseDto;

public interface AdminService {
	void createCategory(Long userId, String name);
	PageResponseDto findFeedbackByPage(Long userId, Pageable pageable);
}
