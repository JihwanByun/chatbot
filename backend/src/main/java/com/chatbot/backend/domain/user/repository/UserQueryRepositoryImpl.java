package com.chatbot.backend.domain.user.repository;

import static com.chatbot.backend.domain.user.entity.QUser.*;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.user.dto.request.UserSearchCondition;
import com.chatbot.backend.domain.user.entity.QUser;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.global.jwt.Role;
import com.chatbot.backend.global.querydsl.Querydsl4RepositorySupport;
import com.querydsl.core.types.dsl.BooleanExpression;

/**
 * 사용자 정보 검색을 위한 QueryDSL 레포지토리 구현체
 * Querydsl4RepositorySupport를 상속받아 페이징 처리를 더 효율적으록 구현
 */
public class UserQueryRepositoryImpl extends Querydsl4RepositorySupport<User, QUser>
	implements UserQueryRepository {

	public UserQueryRepositoryImpl() {
		super(User.class, user);
	}

	/**
	 * 주어진 사용자 검색 조건에 맞는 게시글 목록을 페이지 단위로 조회
	 *
	 * @param userSearchCondition 검색 조건 (레벨, 역할, 사용자 이름, 사용자 부서, 사용자 이메일, 사번)
	 * @param pageable            페이징 정보
	 * @return 조건에 맞는 사용자 페에지
	 */
	@Override
	public Page<User> findAllByConditions(UserSearchCondition userSearchCondition, Pageable pageable) {
		return applyPagination(pageable,
			selectFrom(user)
				.where(
					hasName(userSearchCondition.name()),
					hasEmail(userSearchCondition.email()),
					hasEmployeeNum(userSearchCondition.employeeNum()),
					hasDepartment(userSearchCondition.department()),
					levelEq(userSearchCondition.level()),
					roleEq(userSearchCondition.role())
				)
		);
	}

	// 이름으로 사용자 필터링 (검색 조건에 해당 이름의 일부가 있을 때 조회)
	private BooleanExpression hasName(String name) {
		return name != null ? user.name.contains(name) : null;
	}

	// 이메일로 사용자 필터링 (검색 조건에 해당 이메일의 일부가 있을 때 조회)
	private BooleanExpression hasEmail(String email) {
		return email != null ? user.email.contains(email) : null;
	}

	// 사번으로 사용자 필터링 (검색 조건에 해당 사번의 일부가 있을 때 조회)
	private BooleanExpression hasEmployeeNum(String employeeNum) {
		return employeeNum != null ? user.employeeNum.contains(employeeNum) : null;
	}

	// 부서로 사용자 필터링 (검색 조건에 해당 부서의 일부가 있을 때 조회)
	private BooleanExpression hasDepartment(String department) {
		return department != null ? user.department.contains(department) : null;
	}

	// 레벨로 사용자 필터링 (레벨이 조건과 같을 경우에만 조회)
	private BooleanExpression levelEq(Integer level) {
		return level != null ? user.level.eq(level) : null;
	}

	// 역할로 사용자 필터링 (역할이 조건과 같을 경우에만 조회)
	private BooleanExpression roleEq(Role role) {
		return role != null ? user.role.eq(role) : null;
	}
}
