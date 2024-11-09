package com.chatbot.backend.domain.remark.repository;

import static com.chatbot.backend.domain.board.entity.QBoard.*;
import static com.chatbot.backend.domain.remark.entity.QRemark.*;
import static com.chatbot.backend.domain.user.entity.QUser.*;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.remark.dto.request.RemarkSearchCondition;
import com.chatbot.backend.domain.remark.entity.QRemark;
import com.chatbot.backend.domain.remark.entity.Remark;
import com.chatbot.backend.global.querydsl.Querydsl4RepositorySupport;
import com.querydsl.core.types.dsl.BooleanExpression;

public class RemarkQueryRepositoryImpl extends Querydsl4RepositorySupport<Remark, QRemark>
	implements RemarkQueryRepository {
	public RemarkQueryRepositoryImpl() {
		super(Remark.class, remark);
	}

	@Override
	public Page<Remark> findAllByConditions(Long userId, RemarkSearchCondition remarkSearchCondition,
		Pageable pageable) {
		return applyPagination(pageable,
			select(remark)
				.from(remark)
				.leftJoin(remark.user, user)
				.leftJoin(remark.board, board)
				.where(
					boardIdEq(remarkSearchCondition.boardId()),
					hasContents(remarkSearchCondition.contents()),
					hasName(remarkSearchCondition.name()),
					board.isDeleted.isFalse(),
					remark.isDeleted.isFalse()
				)
		);
	}

	// BoardId로 비고 필터링
	private BooleanExpression boardIdEq(Long boardId) {
		return boardId != null ? remark.board.id.eq(boardId) : null;
	}

	// 내용으로 비고 필터링
	private BooleanExpression hasContents(String contents) {
		return contents != null ? remark.contents.contains(contents) : null;
	}

	// 이름으로 비고 작성자 필터링
	private BooleanExpression hasName(String name) {
		return name != null ? remark.user.name.contains(name) : null;
	}
}
