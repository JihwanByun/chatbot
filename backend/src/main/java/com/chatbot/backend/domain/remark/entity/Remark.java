package com.chatbot.backend.domain.remark.entity;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.global.shared.BaseTimeEntity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Table(name = "Remarks")
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Remark extends BaseTimeEntity {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "remark_id")
	private Long id;

	@Column(nullable = true, columnDefinition = "TEXT")
	private String contents;

	@Column(nullable = false)
	private Boolean isDeleted;

	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "board_id", nullable = false)
	private Board board;

	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "user_id", nullable = false)
	private User user;

	@Builder
	public Remark(String contents, Boolean isDeleted, Board board, User user) {
		this.contents = contents;
		this.isDeleted = isDeleted == null ? false : isDeleted;
		this.board = board;
		this.user = user;
	}

	// Remark 정보 수정
	public void updateRemark(String contents) {
		this.contents = contents;
	}

	// Remark 삭제로 표시
	public void deleteRemark() {
		this.isDeleted = true;
	}
}
