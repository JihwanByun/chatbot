package com.chatbot.backend.domain.category.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class CategoryUnauthorizedException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.CATEGORY_UNAUTHORIZED;

	public CategoryUnauthorizedException() {
		super(errorCode);
	}

	public CategoryUnauthorizedException(Exception exception) {
		super(errorCode, exception);
	}
}