package com.chatbot.backend.domain.admin.service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import com.chatbot.backend.domain.admin.dto.response.ContentDto;
import com.chatbot.backend.domain.admin.dto.response.FeedbackResponseDto;
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
	public FeedbackResponseDto findFeedbackByPage(Long userId, Pageable pageable) {
		// admin 경우에만 feedback 조회 허용
		User user = userRepository.findByIdOrElseThrow(userId);
		if(user.getRole() != Role.ADMIN){
			throw new NoAuthorityException();
		}

		Page<ChatFeedBack> feedBacks = mongoChatFeedBackRepository.findAllByOrderByCreatedAtDesc(pageable);
		List<ContentDto> contents = new ArrayList<>();
		feedBacks.forEach(chatFeedBack -> contents.add(toContentDto(chatFeedBack)));

		return toFeedbackResponseDto(feedBacks, contents);
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

	public ContentDto toContentDto(ChatFeedBack chatFeedBack){
		return
		ContentDto.builder()
			.chatId(String.valueOf(chatFeedBack.getChat().getChatId()))
			.question(chatFeedBack.getChat().getQuestion())
			.answer(chatFeedBack.getChat().getAnswer())
			.isPositive(chatFeedBack.getChat().getIsPositive())
			.negativeReason(chatFeedBack.getNegativeReason())
			.createdAt(LocalDateTime.now())
			.build();
	}

	public FeedbackResponseDto toFeedbackResponseDto(Page<ChatFeedBack> feedBacks, List<ContentDto> contents){
		return
		FeedbackResponseDto.builder()
			.page(feedBacks.getNumber())
			.size(feedBacks.getSize())
			.totalPages(feedBacks.getTotalPages())
			.totalElements(feedBacks.getTotalElements())
			.contents(contents)
			.build();
	}

}
