package com.chatbot.backend.domain.chat.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class FeedBackContradictionException extends ServiceException {

	private static final ErrorCode errorCode = ErrorCode.CONTRADICTION_POSITIVE_OR_NEGATIVE;

	public FeedBackContradictionException() {
		super(errorCode);
	}

	public FeedBackContradictionException(Exception exception) {
		super(errorCode, exception);
	}
}
