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

	public void validateUserAuthroization(User user) {
		if (user.getRole() != Role.ADMIN) {
			throw new CategoryUnauthorizedException();
		}
	}

	public void validateCategoryExists(Category category) {
		if (category.getIsDeleted()) {
			throw new CategoryAlreadyDeletedException();
		}
	}

	public void validateCategoryAlreadyExists(String name) {
		if (categoryRepository.findByName(name).isPresent()) {
			throw new CategoryAlreadyExistsException();
		}
	}
}
