package com.chatbot.backend.domain.admin.service;

import java.util.Optional;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import com.chatbot.backend.domain.admin.dto.response.FeedbackResponseDto;
import com.chatbot.backend.domain.admin.dto.response.PageResponseDto;
import com.chatbot.backend.domain.admin.exception.NoAuthorityException;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.exception.CategoryAlreadyExistsException;
import com.chatbot.backend.domain.category.repository.CategoryRepository;
import com.chatbot.backend.domain.chat.entity.ChatFeedBack;
import com.chatbot.backend.domain.chat.repository.MongoChatFeedBackRepository;
import com.chatbot.backend.domain.chat.repository.MongoChatRepository;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.repository.UserRepository;
import com.chatbot.backend.global.jwt.Role;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
public class AdminServiceImpl implements AdminService {

	private final CategoryRepository categoryRepository;
	private final UserRepository userRepository;
	private final MongoChatFeedBackRepository mongoChatFeedBackRepository;
	private final MongoChatRepository mongoChatRepository;

	@Override
	public PageResponseDto findFeedbackByPage(Long userId, Pageable pageable) {
		// admin 경우에만 feedback 조회 허용
		User user = userRepository.findByIdOrElseThrow(userId);
		if(user.getRole() != Role.ADMIN){
			throw new NoAuthorityException();
		}

		Page<ChatFeedBack> feedBacks = mongoChatFeedBackRepository.findAll(pageable);

		return PageResponseDto.of(feedBacks.map(FeedbackResponseDto::of));
	}

	@Override
	public void createCategory(Long userId, String name) {
		User user = userRepository.findByIdOrElseThrow(userId);
		if(user.getRole() != Role.ADMIN){
			throw new NoAuthorityException();
		}

		Optional<Category> category = categoryRepository.findByName(name);
		if (category.isPresent()) {
			throw new CategoryAlreadyExistsException();
		}

		Category adminCategory = Category.builder()
			.name(name)
			.isDeleted(false)  // isDeleted 기본값 설정
			.build();
		categoryRepository.save(adminCategory);
	}

}
