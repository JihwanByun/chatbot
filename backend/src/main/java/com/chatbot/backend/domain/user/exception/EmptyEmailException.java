package com.chatbot.backend.domain.user.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class EmptyEmailException extends ServiceException {

	private static final ErrorCode errorCode = ErrorCode.EMPTY_EMAIL_EXCEPTION;

	public EmptyEmailException() {
		super(errorCode);
	}

	public EmptyEmailException(Exception exception) {
		super(errorCode, exception);
	}

}
