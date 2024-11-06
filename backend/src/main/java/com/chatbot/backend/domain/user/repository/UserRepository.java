package com.chatbot.backend.domain.user.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.exception.UserNotFoundException;

@Repository
public interface UserRepository extends JpaRepository<User, Long>, UserQueryRepository {

	default User findByIdOrElseThrow(Long userId) {
		return findById(userId).orElseThrow(UserNotFoundException::new);
	}

	default User findByEmailOrElseThrow(String email) {
		return findUserByEmail(email).orElseThrow(UserNotFoundException::new);
	}

	User findUserById(Long id);

	Optional<User> findUserByEmail(String email);
}