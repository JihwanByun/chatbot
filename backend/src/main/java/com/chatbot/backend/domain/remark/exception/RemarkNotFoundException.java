package com.chatbot.backend.domain.remark.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class RemarkNotFoundException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.REMARK_NOT_FOUND;

	public RemarkNotFoundException() {
		super(errorCode);
	}

	public RemarkNotFoundException(Exception exception) {
		super(errorCode, exception);
	}
}