package com.chatbot.backend.domain.board.util;

import org.springframework.stereotype.Component;

import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.exception.BoardAccessDeniedException;
import com.chatbot.backend.domain.board.exception.BoardAlreadyDeletedException;
import com.chatbot.backend.domain.board.exception.BoardCreationNotAuthorizedException;
import com.chatbot.backend.domain.board.exception.BoardLikeAlreadyExistsException;
import com.chatbot.backend.domain.board.exception.BoardUnauthorizedException;
import com.chatbot.backend.domain.board.repository.BoardLikeRepository;
import com.chatbot.backend.domain.remark.entity.Remark;
import com.chatbot.backend.domain.remark.exception.RemarkAlreadyDeletedException;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.global.jwt.Role;

import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class BoardValidator {
	private final BoardLikeRepository boardLikeRepository;

	// 사용자가 게시글 작성 권한이 있는 지 확인
	public void validateUserWriteAuthorized(User user) {
		if (user.getRole() != Role.USER_WRITE) {
			throw new BoardCreationNotAuthorizedException();
		}
	}

	// 사용자가 게시글을 수정 / 삭제할 권한이 있는 지 검증
	public void validateUserAuthorization(User user, Board board) {
		if (!board.isOwnedByUser(user)) {
			throw new BoardUnauthorizedException();
		}
	}

	// 사용자가 게시글에 접근할 권한이 있는 지 확인
	public void validateBoardAccess(User user, Board board) {
		if (!board.isAccessibleByUser(user)) {
			throw new BoardAccessDeniedException();
		}
	}

	// 게시글이 존재하는 지 확인
	public void validateBoardExists(Board board) {
		if (board.getIsDeleted()) {
			throw new BoardAlreadyDeletedException();
		}
	}

	// 중복된 좋아요 여부를 확인
	public void validateBoardLikeExist(Long boardId, Long userId) {
		if (boardLikeRepository.findByBoardIdAndUserId(boardId, userId).isPresent()) {
			throw new BoardLikeAlreadyExistsException();
		}
	}

	// 비고가 존재하는 지 확인
	public void validateRemarkExists(Remark remark) {
		if (remark.getIsDeleted()) {
			throw new RemarkAlreadyDeletedException();
		}
	}
}
