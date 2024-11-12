package com.chatbot.backend.domain.category.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.exception.CategoryNotFoundException;

@Repository
public interface CategoryRepository extends JpaRepository<Category, Long> {
	Optional<Category> findByName(String name);

	Optional<Category> findByNameAndIsDeletedFalse(String name);

	List<Category> findAllByIsDeletedFalse();

	default Category findByNameOrElseThrow(String name) {
		return this.findByNameAndIsDeletedFalse(name).orElseThrow(CategoryNotFoundException::new);
	}

	default Category findByIdOrElseThrow(Long id) {
		return findById(id).orElseThrow(CategoryNotFoundException::new);
	}
}
