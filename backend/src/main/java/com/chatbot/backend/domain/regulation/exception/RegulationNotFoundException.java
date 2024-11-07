package com.chatbot.backend.domain.regulation.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class RegulationNotFoundException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.REGULATION_NOT_FOUND;

	public RegulationNotFoundException() {
		super(errorCode);
	}

	public RegulationNotFoundException(Exception exception) {
		super(errorCode, exception);
	}
}