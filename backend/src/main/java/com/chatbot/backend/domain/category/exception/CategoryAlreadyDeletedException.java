package com.chatbot.backend.domain.category.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class CategoryAlreadyDeletedException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.CATEGORY_ALREADY_DELETED;

	public CategoryAlreadyDeletedException() {
		super(errorCode);
	}

	public CategoryAlreadyDeletedException(Exception exception) {
		super(errorCode, exception);
	}
}