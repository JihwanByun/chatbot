package com.chatbot.backend.domain.remark.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class RemarkAlreadyDeletedException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.REMARK_ALREADY_DELETED;

	public RemarkAlreadyDeletedException() {
		super(errorCode);
	}

	public RemarkAlreadyDeletedException(Exception exception) {
		super(errorCode, exception);
	}
}