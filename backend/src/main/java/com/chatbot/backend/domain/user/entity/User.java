package com.chatbot.backend.domain.user.entity;

import java.util.ArrayList;
import java.util.List;

import com.chatbot.backend.domain.chatroom.entity.ChatRoom;
import com.chatbot.backend.domain.user.dto.request.UserUpdateRequestDto;
import com.chatbot.backend.global.jwt.Role;
import com.chatbot.backend.global.shared.BaseTimeEntity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "users")
@Getter
@NoArgsConstructor
public class User extends BaseTimeEntity {

	@Builder
	public User(String email, String password, String name, String employeeNum, String department, Integer level,
		String profileUrl, Boolean isDeleted, Role role) {
		this.email = email;
		this.password = password;
		this.name = name;
		this.employeeNum = employeeNum;
		this.department = department;
		this.level = level;
		this.profileUrl = profileUrl;
		this.isDeleted = isDeleted;
		this.role = role;
	}

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "USER_ID")
	private Long id;

	@Column(nullable = false, unique = true, length = 100)
	private String email;

	@Column(nullable = false, length = 255)
	private String password;

	@Column(nullable = false, length = 50)
	private String name;

	@Column(name = "employee_num", nullable = false, length = 255)
	private String employeeNum;

	@Column(nullable = false, length = 100)
	private String department;

	@Column(nullable = false)
	private int level;

	@Column(name = "profile_url", nullable = false, length = 2000)
	private String profileUrl;

	@Column(name = "is_deleted", nullable = false)
	private boolean isDeleted;

	@Column(name = "can_create_board", nullable = false)
	private boolean canCreateBoard;

	@OneToMany(fetch = FetchType.LAZY, mappedBy = "user")
	private List<ChatRoom> chatRoom = new ArrayList<>();

	// 지연로딩 : 필요할때만 가져온다.
	@Enumerated(EnumType.STRING)
	@Column(nullable = false)
	private Role role;

	// User 정보 수정
	public void updateUser(UserUpdateRequestDto userUpdateRequestDto) {
		this.level = userUpdateRequestDto.level() != null ? userUpdateRequestDto.level() : level;
		this.role = userUpdateRequestDto.role() != null ? userUpdateRequestDto.role() : role;
	}

	// User 정보 삭제
	public void deleteUser(User user) {
		this.isDeleted = true;
	}
}
