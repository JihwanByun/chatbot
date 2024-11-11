package com.chatbot.backend.domain.remark.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.board.util.BoardValidator;
import com.chatbot.backend.domain.remark.dto.request.RemarkCreateRequestDto;
import com.chatbot.backend.domain.remark.dto.request.RemarkSearchCondition;
import com.chatbot.backend.domain.remark.dto.request.RemarkUpdateRequestDto;
import com.chatbot.backend.domain.remark.dto.response.RemarkBaseResponseDto;
import com.chatbot.backend.domain.remark.dto.response.RemarkDetailResponseDto;
import com.chatbot.backend.domain.remark.entity.Remark;
import com.chatbot.backend.domain.remark.repository.RemarkRepository;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.repository.UserRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = false)
public class RemarkServiceImpl implements RemarkService {
	private final RemarkRepository remarkRepository;
	private final UserRepository userRepository;
	private final BoardRepository boardRepository;
	private final BoardValidator boardValidator;

	@Override
	public RemarkDetailResponseDto createRemark(Long userId,
		RemarkCreateRequestDto remarkCreateRequestDto) {
		// 조회 & 검증
		User user = userRepository.findByIdOrElseThrow(userId);
		Board board = boardRepository.findByIdOrElseThrow(remarkCreateRequestDto.boardId());

		// 검증
		boardValidator.validateBoardExists(board);
		boardValidator.validateUserWriteAuthorized(user);

		// 비고 생성 (비즈니스 로직)
		Remark remark = remarkCreateRequestDto.toDocument(user, board);
		remarkRepository.save(remark);
		return RemarkDetailResponseDto.of(remark);
	}

	@Override
	public RemarkDetailResponseDto updateRemark(Long userId, Long remarkId,
		RemarkUpdateRequestDto remarkUpdateRequestDto) {
		// 조회 & 검증
		boardValidator.validateUserWriteAuthorized(userRepository.findByIdOrElseThrow(userId));

		Remark remark = remarkRepository.findByIdOrElseThrow(remarkId);
		boardValidator.validateRemarkExists(remark);

		// 비고 업데이트 (비즈니스 로직)
		remark.updateRemark(remarkUpdateRequestDto.contents());
		return RemarkDetailResponseDto.of(remark);
	}

	@Override
	public void deleteRemark(Long userId, Long remarkId) {
		// 조회 & 검증
		Remark remark = remarkRepository.findByIdOrElseThrow(remarkId);

		// 검증
		boardValidator.validateUserWriteAuthorized(userRepository.findByIdOrElseThrow(userId));
		boardValidator.validateRemarkExists(remark);

		// 비고 삭제 (비즈니스 로직)
		remark.deleteRemark();
	}

	@Override
	public RemarkDetailResponseDto getRemark(Long userId, Long remarkId) {
		// 조회 & 검증
		Remark remark = remarkRepository.findByIdOrElseThrow(remarkId);

		// 검증
		boardValidator.validateRemarkExists(remark);

		return RemarkDetailResponseDto.of(remark);
	}

	@Override
	public Page<RemarkBaseResponseDto> getRemarkList(Long userId, RemarkSearchCondition remarkSearchCondition,
		Pageable pageable) {
		return remarkRepository.findAllByConditions(userId, remarkSearchCondition, pageable)
			.map(RemarkBaseResponseDto::of);
	}
}
