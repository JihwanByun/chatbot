package com.chatbot.backend.domain.category.util;

import org.springframework.stereotype.Component;

import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.exception.CategoryAlreadyDeletedException;
import com.chatbot.backend.domain.category.exception.CategoryAlreadyExistsException;
import com.chatbot.backend.domain.category.exception.CategoryUnauthorizedException;
import com.chatbot.backend.domain.category.repository.CategoryRepository;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.global.jwt.Role;

import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class CategoryValidator {
	private final CategoryRepository categoryRepository;

	// 사용자 권한 검증
	public void validateUserAuthroization(User user) {
		if (user.getRole() != Role.ADMIN) {
			throw new CategoryUnauthorizedException();
		}
	}

	// 카테고리 존재 여부 검증
	public void validateCategoryExists(Category category) {
		if (category.getIsDeleted()) {
			throw new CategoryAlreadyDeletedException();
		}
	}

	// 카테고리 이름 중복 여부 검증
	public void validateCategoryAlreadyExists(String name) {
		if (categoryRepository.findByNameAndIsDeletedFalse(name).isPresent()) {
			throw new CategoryAlreadyExistsException();
		}
	}
}
