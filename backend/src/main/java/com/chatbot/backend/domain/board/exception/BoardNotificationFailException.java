package com.chatbot.backend.domain.board.exception;

import com.chatbot.backend.global.error.ErrorCode;
import com.chatbot.backend.global.error.ServiceException;

public class BoardNotificationFailException extends ServiceException {
	private static final ErrorCode errorCode = ErrorCode.BOARD_NOTIFICATION_FAIL;

	public BoardNotificationFailException() {
		super(errorCode);
	}

	public BoardNotificationFailException(Exception exception) {
		super(errorCode, exception);
	}
}