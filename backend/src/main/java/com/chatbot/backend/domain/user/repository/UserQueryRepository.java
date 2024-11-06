package com.chatbot.backend.domain.user.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.user.dto.request.UserSearchCondition;
import com.chatbot.backend.domain.user.entity.User;

public interface UserQueryRepository {
	Page<User> findAllByConditions(UserSearchCondition userSearchCondition, Pageable pageable);
}
