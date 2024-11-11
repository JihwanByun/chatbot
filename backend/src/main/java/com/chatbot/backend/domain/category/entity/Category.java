package com.chatbot.backend.domain.category.entity;

import org.hibernate.annotations.ColumnDefault;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Table(name = "Categories")
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString
public class Category {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "category_id")
	private Long id;

	@Column(nullable = false)
	private String name;

	@ColumnDefault("false")
	private Boolean isDeleted;

	@Builder
	public Category(String name, Boolean isDeleted) {
		this.name = name;
		this.isDeleted = isDeleted == null ? false : isDeleted;
	}

	// Category 수정
	public void updateCategory(String name) {
		this.name = name;
	}

	// Category 삭제
	public void deleteCategory() {
		this.isDeleted = true;
	}
}
