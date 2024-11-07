package com.chatbot.backend.domain.admin.service;

import java.util.Optional;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.chatbot.backend.domain.admin.dto.response.FeedbackResponseDto;
import com.chatbot.backend.domain.admin.dto.response.PageResponseDto;
import com.chatbot.backend.domain.admin.exception.NoAuthorityException;
import com.chatbot.backend.domain.category.dto.response.CategoryItemDto;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.exception.CategoryAlreadyExistsException;
import com.chatbot.backend.domain.category.repository.CategoryRepository;
import com.chatbot.backend.domain.category.util.CategoryValidator;
import com.chatbot.backend.domain.chat.entity.ChatFeedBack;
import com.chatbot.backend.domain.chat.repository.MongoChatFeedBackRepository;
import com.chatbot.backend.domain.chat.repository.MongoChatRepository;
import com.chatbot.backend.domain.user.dto.request.UserSearchCondition;
import com.chatbot.backend.domain.user.dto.request.UserUpdateRequestDto;
import com.chatbot.backend.domain.user.dto.response.UserBaseResponseDto;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.repository.UserRepository;
import com.chatbot.backend.global.jwt.Role;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional(readOnly = false)
public class AdminServiceImpl implements AdminService {

	private final CategoryRepository categoryRepository;
	private final UserRepository userRepository;
	private final MongoChatFeedBackRepository mongoChatFeedBackRepository;
	private final MongoChatRepository mongoChatRepository;
	private final CategoryValidator categoryValidator;

	//== ADMIN Feedback 관련 코드 ==//
	@Override
	public PageResponseDto findFeedbackByPage(Long userId, Boolean isPositive, Pageable pageable) {
		// admin 경우에만 feedback 조회 허용
		User user = userRepository.findByIdOrElseThrow(userId);
		if (user.getRole() != Role.ADMIN) {
			throw new NoAuthorityException();
		}

		Page<ChatFeedBack> feedBacks = (isPositive == null) ? mongoChatFeedBackRepository.findAll(pageable)
			: (isPositive) ? mongoChatFeedBackRepository.findAllByPositiveFeedback(pageable) : mongoChatFeedBackRepository.findAllByNegativeFeedback(pageable);

		return PageResponseDto.of(feedBacks.map(FeedbackResponseDto::of));
	}

	//== ADMIN Category 관련 코드 ==//
	@Override
	public void createCategory(Long userId, String name) {
		User user = userRepository.findByIdOrElseThrow(userId);
		if (user.getRole() != Role.ADMIN) {
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

	/**
	 * 카테고리 수정
	 *
	 * @param userId     사용자 ID
	 * @param categoryId 삭제할 카테고리 ID
	 * @param name       수정될 카테고리 이름
	 * @return 수정된 카테고리 Dto
	 */
	@Override
	public CategoryItemDto updateCategory(Long userId, Long categoryId, String name) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Category category = categoryRepository.findByIdOrElseThrow(categoryId);

		// 검증
		categoryValidator.validateUserAuthroization(user);
		categoryValidator.validateCategoryExists(category);
		categoryValidator.validateCategoryAlreadyExists(name);

		// 카테고리 수정 처리
		category.updateCategory(name);
		return CategoryItemDto.of(category);
	}

	/**
	 * 카테고리 삭제
	 *
	 * @param userId     사용자 ID
	 * @param categoryId 삭제할 카테고리 ID
	 */
	@Override
	public void deleteCategory(Long userId, Long categoryId) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Category category = categoryRepository.findByIdOrElseThrow(categoryId);

		// 검증
		categoryValidator.validateUserAuthroization(user);
		categoryValidator.validateCategoryExists(category);

		// 카테고리 삭제 처리
		category.deleteCategory();
	}

	//== ADMIN User 관련 코드 ==//

	/**
	 * 사용자 정보 수정
	 *
	 * @param adminId              로그인한 사용자 ID
	 * @param userId               수정할 사용자 ID
	 * @param userUpdateRequestDto 수정할 사용자 정보 (LEVEL, ROLE)
	 * @return 수정 후 수정된 사용자 정보
	 */
	@Override
	public UserBaseResponseDto updateUser(Long adminId, Long userId, UserUpdateRequestDto userUpdateRequestDto) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		User admin = userRepository.findByIdOrElseThrow(adminId);

		// 검증
		categoryValidator.validateUserAuthroization(admin);

		// 유저 정보 업데이트 처리
		user.updateUser(userUpdateRequestDto);
		return UserBaseResponseDto.of(user);
	}

	/**
	 * 사용자 정보 목록 조회
	 *
	 * @param userId              로그인한 사용자 ID
	 * @param userSearchCondition 사용자 검색 조건
	 * @param pageable
	 * @return 사용자 정보를 담은 페이징 객체
	 */
	@Override
	@Transactional(readOnly = true)
	public Page<UserBaseResponseDto> getUserList(Long userId, UserSearchCondition userSearchCondition,
		Pageable pageable) {
		return userRepository.findAllByConditions(userSearchCondition, pageable).map(UserBaseResponseDto::of);
	}
}
