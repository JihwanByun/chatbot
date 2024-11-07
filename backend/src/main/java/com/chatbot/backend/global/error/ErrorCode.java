package com.chatbot.backend.global.error;

import lombok.AllArgsConstructor;
import lombok.Getter;

// 사용되는 에러 코드를 정의하는 열거형 클래스
@Getter
@AllArgsConstructor
public enum ErrorCode {
	// 서버 오류
	INTERNAL_SERVER_ERROR(500, "INTERNAL_SERVER_ERROR", "서버 오류가 발생했습니다."),

	// 공통 오류
	BAD_REQUEST(400, "BAD_REQUEST", "잘못된 요청입니다."),
	UNAUTHORIZED(401, "UNAUTHORIZED", "인증이 필요합니다."),
	FORBIDDEN(403, "FORBIDDEN", "권한이 없습니다."),
	NOT_FOUND(404, "NOT_FOUND", "리소스를 찾을 수 없습니다."),
	CONFLICT(409, "CONFLICT", "리소스 충돌이 발생했습니다."),

	// JWT 관련 오류
	INVALID_TOKEN(422, "INVALID_TOKEN", "유효하지 않은 토큰입니다."),
	EXPIRED_TOKEN(401, "EXPIRED_TOKEN", "토큰이 만료되었습니다."),
	UNSUPPORTED_TOKEN(400, "UNSUPPORTED_TOKEN", "지원되지 않는 토큰 형식입니다."),
	NO_REFRESH_TOKEN(404, "NO_REFRESH_TOKEN", "토큰이 없습니다."),

	// 회원가입 관련 오류
	DUPLICATE_EMAIL(409, "DUPLICATE_EMAIL", "이미 존재하는 이메일입니다."),

	// 게시글 관련 오류
	BOARD_NOT_FOUND(404, "BOARD_NOT_FOUND", "해당 게시글을 찾을 수 없습니다."),
	BOARD_ALREADY_DELETED(400, "BOARD_ALREADY_DELETED", "이미 삭제된 게시글입니다."),
	BOARD_ACCESS_DENIED(403, "BOARD_ACCESS_DENIED", "게시글에 대한 접근 권한이 없습니다."),
	BOARD_LEVEL_INSUFFICIENT(403, "BOARD_LEVEL_INSUFFICIENT", "게시글 열람 권한 레벨이 부족합니다."),
	BOARD_CREATION_NOT_AUTHORIZED(403, "BOARD_CREATION_NOT_AUTHORIZED", "게시글 작성 권한이 없습니다."),

	// 게시글 내용 관련 오류
	BOARD_TITLE_EMPTY(400, "BOARD_TITLE_EMPTY", "게시글 제목은 필수입니다."),
	BOARD_TITLE_TOO_LONG(400, "BOARD_TITLE_TOO_LONG", "게시글 제목이 너무 깁니다. (최대 200자)"),
	BOARD_CATEGORY_NOT_FOUND(404, "BOARD_CATEGORY_NOT_FOUND", "존재하지 않는 게시글 카테고리입니다."),
	BOARD_CATEGORY_DELETED(400, "BOARD_CATEGORY_DELETED", "삭제된 카테고리입니다."),

	// 파일 관련 오류
	FILE_UPLOAD_FAILED(500, "FILE_UPLOAD_FAILED", "파일 업로드에 실패했습니다."),
	FILE_DOWNLOAD_FAILED(500, "FILE_DOWNLOAD_FAILED", "파일 다운로드에 실패했습니다."),
	FILE_DIRECTORY_INVALID(400, "FILE_DIRECTORY_INVALID", "잘못된 파일 디렉토리입니다."),
	FILE_NOT_FOUND(400, "FILE_NOT_FOUND", "존재하지 않는 파일입니다."),
	BOARD_FILE_SIZE_EXCEEDED(400, "BOARD_FILE_SIZE_EXCEEDED", "파일 크기가 제한을 초과했습니다."),
	BOARD_FILE_TYPE_NOT_SUPPORTED(400, "BOARD_FILE_TYPE_NOT_SUPPORTED", "지원하지 않는 파일 형식입니다."),
	BOARD_FILE_URL_INVALID(400, "BOARD_FILE_URL_INVALID", "잘못된 파일 URL입니다."),
	BOARD_NULL(400, "BOARD_NULL", "게시판의 값이 NULL 입니다."),

	// 게시글 수정 삭제 관련 오류
	BOARD_UNAUTHORIZED(403, "BOARD_UNAUTHORIZED", "게시글 수정 / 삭제 권한이 없습니다."),

	// 게시글 좋아요 관련 오류
	BOARD_LIKE_ALREADY_EXISTS(409, "BOARD_LIKE_ALREADY_EXISTS", "이미 좋아요를 누른 게시글입니다."),
	BOARD_LIKE_NOT_FOUND(404, "BOARD_LIKE_NOT_FOUND", "존재하지 않는 좋아요입니다."),

	// 사용자 관련 오류
	USER_NOT_FOUND(404, "USER_NOT_FOUND", "사용자를 찾을 수 없습니다."),
	USER_DELETED(400, "USER_DELETED", "삭제된 사용자입니다."),
	USER_NULL(400, "USER_NULL", "사용자의 값이 NULL 입니다."),
	NO_AUTHORITY(403, "NO_AUTHORITY", "권한이 없습니다"),

	// 카테고리 관련 오류
	CATEGORY_NOT_FOUND(404, "CATEGORY_NOT_FOUND", "카테고리를 찾을 수 없습니다."),
	CATEGORY_ALREADY_EXISTS(409, "CATEGORY_ALREADY_EXISTS", "이미 존재하는 카테고리입니다."),
	CATEGORY_ALREADY_DELETED(400, "CATEGORY_ALREADY_DELETED", "이미 삭제된 카테고리입니다."),
	CATEGORY_UNAUTHORIZED(403, "CATEGORY_UNAUTHORIZED", "카테고리 수정 / 삭제 권한이 없습니다."),

	// 로그인 관련 오류
	INCORRECT_ID_OR_PASSWORD(401, "INCORRECT_ID_OR_PASSWORD", "아이디 혹은 비밀번호가 틀렸습니다."),

	//채팅 관련 오류
	CHAT_NOT_FOUND(404, "CHAT_ID_NOT_FOUND", "채팅을 찾을 수 없습니다"),

	//채팅 피드백 관련 오류
	CONTRADICTION_POSITIVE_OR_NEGATIVE(404, "CAN'T_CREATE_POSITIVE_FEEDBACK", "긍정 피드백에 부정 피드백을 작성할 수 없습니다.");

	private final int httpStatus;
	private final String code;
	private final String message;
}
