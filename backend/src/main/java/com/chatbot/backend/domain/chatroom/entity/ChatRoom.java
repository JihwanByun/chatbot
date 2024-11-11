package com.chatbot.backend.domain.chatroom.entity;

import java.time.LocalDateTime;

import org.hibernate.annotations.ColumnDefault;

import com.chatbot.backend.domain.user.entity.User;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@Table(name = "chatrooms")
@NoArgsConstructor()
public class ChatRoom {

	@Id
	@GeneratedValue
	@Column(name = "CHATROOM_ID")
	private Long id;

	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name = "USERS_ID")
	private User user;

	private String title;
	private LocalDateTime createdAt;
	@Column(nullable = true)
	private LocalDateTime modifiedAt;

	@ColumnDefault("false")
	private boolean isDeleted;

	@Builder
	public ChatRoom(User user, String content) {
		this.user = user;
		this.title = content.length() >= 8 ? content.substring(0, 8) : content;
		this.createdAt = LocalDateTime.now();
	}

	public void setDeleted(boolean deleted) {
		this.isDeleted = deleted;
	}

}
