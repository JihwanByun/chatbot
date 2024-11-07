package com.chatbot.backend.domain.board.repository;

import static com.chatbot.backend.domain.board.entity.QBoard.*;
import static com.chatbot.backend.domain.board.entity.QBoardLike.*;
import static com.chatbot.backend.domain.user.entity.QUser.*;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.entity.QBoard;
import com.chatbot.backend.global.querydsl.Querydsl4RepositorySupport;
import com.querydsl.core.types.dsl.BooleanExpression;

/**
 * 게시판 검색을 위한 QueryDSL 리포지토리 구현체
 * Querydsl4RepositorySupport를 상속받아 페이징 처리를 더 효율적으로 구현
 */
public class BoardQueryRepositoryImpl extends Querydsl4RepositorySupport<Board, QBoard>
	implements BoardQueryRepository {

	// 생성자에서 Board 엔티티와 QBoard 객체 설정
	public BoardQueryRepositoryImpl() {
		super(Board.class, board);
	}

	/**
	 * 주어진 검색 조건에 맞는 게시글 목록을 페이지 단위로 조회
	 * @param userId 사용자 ID
	 * @param boardSearchCondition 검색 조건 (레벨, 작성자 이름, 제목)
	 * @param pageable 페이징 정보
	 * @return 조건에 맞는 게시글 페이지
	 */
	@Override
	public Page<Board> findAllByConditions(Long userId, BoardSearchCondition boardSearchCondition,
		Pageable pageable) {
		return applyPagination(pageable,
			select(board)
				.from(board)
				.leftJoin(board.user, user)
				.leftJoin(boardLike).on(boardLike.board.eq(board).and(boardLike.user.id.eq(userId)))
				.where(
					isAccessibleByLevel(boardSearchCondition.level()),
					hasName(boardSearchCondition.name()),
					hasTitle(boardSearchCondition.title()),
					board.isDeleted.isFalse()));
	}

	// 게시글 접근 권한 확인을 위한 메소드
	// 사용자 레벨에 따른 접근 가능 여부 판단
	private BooleanExpression isAccessibleByLevel(Integer level) {
		return level != null ? board.level.eq(level).and(board.level.loe(user.level)) : board.level.loe(user.level);
	}

	// 이름으로 게시글 작성자 필터링
	private BooleanExpression hasName(String name) {
		return name != null ? board.user.name.contains(name) : null;
	}

	// 제목으로 게시글 필터링
	private BooleanExpression hasTitle(String title) {
		return title != null ? board.title.contains(title) : null;
	}
}