package com.chatbot.backend.domain.regulation.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class RegulationInvalidException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.INVALID_REGULATION;

	public RegulationInvalidException() {
		super(errorCode);
	}

	public RegulationInvalidException(Exception exception) {
		super(errorCode, exception);
	}
}
