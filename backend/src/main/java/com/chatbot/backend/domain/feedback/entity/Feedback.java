package com.chatbot.backend.domain.feedback.entity;

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

@Table(name = "Feedbacks")
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Feedback {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "feedback_id")
	private Long id;

	@Column(nullable = false)
	private String name;

	private Boolean isDeleted;

	@Builder
	public Feedback(String name, Boolean isDeleted) {
		this.name = name;
		this.isDeleted = isDeleted == null ? false : isDeleted;
	}

	// Feedback 삭제
	public void deleteFeedback() {
		this.isDeleted = true;
	}
}
