package com.chatbot.backend.domain.board.repository;

import java.util.Optional;

import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Slice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.board.entity.BoardLike;
import com.chatbot.backend.domain.board.exception.BoardLikeNotFoundException;

@Repository
public interface BoardLikeRepository extends JpaRepository<BoardLike, Long> {
	Optional<BoardLike> findByBoardIdAndUserId(Long boardId, Long userId);

	default BoardLike findByBoardIdAndUserIdOrElseThrow(Long boardId, Long userId) {
		return findByBoardIdAndUserId(boardId, userId).orElseThrow(BoardLikeNotFoundException::new);
	}

	Slice<BoardLike> findByUserId(Long userId, Pageable pageable);
}
