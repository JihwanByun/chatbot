package com.chatbot.backend.domain.category.service;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.chatbot.backend.domain.category.dto.response.CategoryFindResponseDto;
import com.chatbot.backend.domain.category.dto.response.CategoryItemDto;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.repository.CategoryRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional(readOnly = true)
public class CategoryServiceImpl implements CategoryService {

	private final CategoryRepository categoryRepository;

	@Override
	public CategoryFindResponseDto getCategoryList() {
		List<Category> categories = categoryRepository.findAllByIsDeletedFalse();

		List<CategoryItemDto> categoryItemList =
			categories.stream()
				.map(category -> new CategoryItemDto(category.getId(), category.getName().trim()))
				.toList();

		return new CategoryFindResponseDto(categoryItemList);
	}
}
