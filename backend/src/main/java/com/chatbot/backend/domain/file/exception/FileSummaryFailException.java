package com.chatbot.backend.domain.file.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class FileSummaryFailException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.FILE_SUMMARY_FAIL;

	public FileSummaryFailException() {
		super(errorCode);
	}

	public FileSummaryFailException(Exception exception) {
		super(errorCode, exception);
	}
}