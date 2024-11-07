package com.chatbot.backend.domain.board.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.entity.Board;

public interface BoardQueryRepository {
	Page<Board> findAllByConditions(Long userId, BoardSearchCondition boardSearchCondition,
		Pageable pageable);
}
