package com.chatbot.backend.global.querydsl;

import java.util.List;
import java.util.function.Function;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.support.JpaEntityInformation;
import org.springframework.data.jpa.repository.support.JpaEntityInformationSupport;
import org.springframework.data.jpa.repository.support.Querydsl;
import org.springframework.data.querydsl.SimpleEntityPathResolver;
import org.springframework.data.support.PageableExecutionUtils;
import org.springframework.stereotype.Component;
import org.springframework.util.Assert;

import com.querydsl.core.types.EntityPath;
import com.querydsl.core.types.Expression;
import com.querydsl.core.types.dsl.EntityPathBase;
import com.querydsl.core.types.dsl.PathBuilder;
import com.querydsl.jpa.impl.JPAQuery;
import com.querydsl.jpa.impl.JPAQueryFactory;

import jakarta.persistence.EntityManager;

/**
 * Querydsl 기반의 커스텀 리포지토리 지원
 * 페이징 처리를 위한 메서드 제공
 * 효율적인 쿼리 작성과 결과 페이징
 */
@Component
public abstract class Querydsl4RepositorySupport<U, Q extends EntityPathBase<U>> {
	// 도메인 클래스 타입
	private final Class<U> domainClass;
	// QueryDSL 엔티티 객체
	private final Q qObject;
	// QueryDSL 관련 객체들
	private Querydsl querydsl;
	private EntityManager entityManager;
	private JPAQueryFactory queryFactory;

	/**
	 * 도메인 클래스와 QueryDSL 엔티티 경로 객체를 설정하는 생성자
	 * @param domainClass 도메인 클래스의 타입
	 * @param qEntity QueryDSL 엔티티 경로 객체
	 */
	public Querydsl4RepositorySupport(Class<U> domainClass, Q qEntity) {
		Assert.notNull(domainClass, "Domain class must not be null.");
		Assert.notNull(qEntity, "Q Entity must not be null.");
		this.domainClass = domainClass;
		this.qObject = qEntity;
	}

	/**
	 * EntityManager를 주입하고, QueryDSL 객체를 초기화하는 메서드
	 * @param entityManager JPA EntityManager
	 */
	@Autowired
	public void setEntityManager(EntityManager entityManager) {
		// 도메인 클래스 정보를 기반으로 JPA 엔티티 정보를 가져옴
		JpaEntityInformation<U, ?> entityInformation = JpaEntityInformationSupport.getEntityInformation(domainClass,
			entityManager);
		SimpleEntityPathResolver resolver = SimpleEntityPathResolver.INSTANCE;

		// QueryDSL 엔티티 경로를 생성
		EntityPath<U> path = resolver.createPath(entityInformation.getJavaType());
		this.entityManager = entityManager;

		// QueryDSL 설정을 위한 엔티티 매니저와 경로 빌더로 QueryDSL 객체 생성
		this.querydsl = new Querydsl(entityManager, new PathBuilder<>(path.getType(), path.getMetadata()));
	}

	/**
	 * JPAQueryFactory를 주입받아 쿼리 생성을 지원
	 * @param jpaQueryFactory QueryDSL을 위한 JPAQueryFactory
	 */
	@Autowired
	public void setQueryFactory(JPAQueryFactory jpaQueryFactory) {
		this.queryFactory = jpaQueryFactory;
	}

	// JPAQueryFactory 반환
	protected JPAQueryFactory getQueryFactory() {
		return queryFactory;
	}

	// QueryDSL 객체 반환
	protected Querydsl getQuerydsl() {
		return querydsl;
	}

	// EntityManager 반환
	protected EntityManager getEntityManager() {
		return entityManager;
	}

	/**
	 * 특정 표현식으로 선택 쿼리를 생성
	 * @param expr 조회할 데이터 표현식
	 * @param <T> 조회할 데이터 타입
	 * @return JPAQuery 쿼리 객체
	 */
	protected <T> JPAQuery<T> select(Expression<T> expr) {
		return getQueryFactory().select(expr);
	}

	/**
	 * 특정 엔티티로부터 데이터를 선택하는 쿼리를 생성
	 * @param from 데이터를 조회할 엔티티 경로
	 * @param <T> 조회할 데이터 타입
	 * @return JPAQuery 쿼리 객체
	 */
	protected <T> JPAQuery<T> selectFrom(EntityPath<T> from) {
		return getQueryFactory().selectFrom(from);
	}

	/**
	 * 여러 엔티티로부터 데이터를 조회하는 쿼리를 생성
	 * @param from 조회할 엔티티 경로들
	 * @return JPAQuery 쿼리 객체
	 */
	protected JPAQuery<?> from(EntityPath<?>... from) {
		return getQuerydsl().createQuery().from(from);
	}

	/**
	 * 페이징 처리와 함께 쿼리를 실행하여 결과를 반환
	 * @param pageable 페이징 정보
	 * @param contentQuery 데이터 조회를 위한 함수
	 * @param countQuery 카운트 조회를 위한 함수
	 * @param <T> 데이터 타입
	 * @return 페이징 처리된 결과 페이지
	 */
	protected <T> Page<T> applyPagination(Pageable pageable, Function<JPAQueryFactory, JPAQuery<T>> contentQuery,
		Function<JPAQueryFactory, JPAQuery<Long>> countQuery) {

		// 페이징 처리를 적용하여 조회할 데이터 쿼리 실행
		JPAQuery<T> jpaContentQuery = contentQuery.apply(getQueryFactory());
		List<T> content = getQuerydsl().applyPagination(pageable, jpaContentQuery).fetch();

		// 총 개수를 조회하는 쿼리 실행
		JPAQuery<Long> countResultQuery = countQuery.apply(getQueryFactory());

		// 페이징 결과를 생성하여 반환
		return PageableExecutionUtils.getPage(content, pageable, countResultQuery::fetchOne);
	}

	/**
	 * 페이징 처리와 함께 쿼리를 실행하며, 별도의 데이터와 카운트 쿼리로 분리
	 * @param pageable 페이징 정보
	 * @param contentQuery 데이터 조회 쿼리
	 * @param countQuery 카운트 조회 쿼리
	 * @param <T> 데이터 타입
	 * @return 페이징 처리된 결과 페이지
	 */
	protected <T> Page<T> applyPagination(Pageable pageable, JPAQuery<T> contentQuery, JPAQuery<Long> countQuery) {
		// 페이징 처리를 적용하여 데이터 쿼리 실행
		List<T> content = getQuerydsl().applyPagination(pageable, contentQuery).fetch();

		// 페이징 객체를 생성하여 반환
		return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
	}

	/**
	 * 기본 페이징 쿼리로 총 개수와 결과를 조회하여 페이징 처리
	 * @param pageable 페이징 정보
	 * @param contentQuery 데이터 조회 쿼리
	 * @param <T> 데이터 타입
	 * @return 페이징 처리된 결과 페이지
	 */
	protected <T> Page<T> applyPagination(Pageable pageable, JPAQuery<T> contentQuery) {
		// 총 개수를 계산하는 쿼리 생성
		JPAQuery<Long> countQuery = contentQuery.clone().select(qObject.count());
		return applyPagination(pageable, contentQuery, countQuery);
	}

	/**
	 * 특정 표현식으로 데이터와 카운트 쿼리를 구성하여 페이징 처리
	 * @param pageable 페이징 정보
	 * @param contentExpr 데이터 조회 표현식
	 * @param countExpr 카운트 조회 표현식
	 * @param baseQuery 기본 쿼리
	 * @param <T> 데이터 타입
	 * @return 페이징 처리된 결과 페이지
	 */
	protected <T> Page<T> applyPagination(Pageable pageable, Expression<T> contentExpr, Expression<Long> countExpr,
		JPAQuery<?> baseQuery) {
		// 조회할 데이터를 위한 쿼리 생성
		JPAQuery<T> contentQuery = baseQuery.clone().select(contentExpr);

		// 총 개수를 계산하기 위한 쿼리 생성
		JPAQuery<Long> countQuery = baseQuery.clone().select(countExpr);
		return applyPagination(pageable, contentQuery, countQuery);
	}
}