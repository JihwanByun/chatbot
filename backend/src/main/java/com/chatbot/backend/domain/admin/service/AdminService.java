package com.chatbot.backend.domain.admin.service;

import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.admin.dto.response.PageResponseDto;
import com.chatbot.backend.domain.category.dto.response.CategoryItemDto;

public interface AdminService {
	void createCategory(Long userId, String name);

	PageResponseDto findFeedbackByPage(Long userId, Pageable pageable);

	CategoryItemDto updateCategory(Long userId, Long categoryId, String name);

	void deleteCategory(Long userId, Long categoryId);
}
