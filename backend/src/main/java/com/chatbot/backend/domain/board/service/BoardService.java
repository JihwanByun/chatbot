package com.chatbot.backend.domain.board.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.board.dto.request.BoardCreateRequest;
import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.dto.request.BoardUpdateRequest;
import com.chatbot.backend.domain.board.dto.response.BoardBaseResponse;
import com.chatbot.backend.domain.board.dto.response.BoardDetailResponse;

public interface BoardService {
	BoardDetailResponse createBoard(Long userId, BoardCreateRequest boardCreateRequest, MultipartFile file);

	BoardDetailResponse updateBoard(Long userId, Long boardId, BoardUpdateRequest boardUpdateRequest,
		MultipartFile file);

	void deleteBoard(Long userId, Long boardId);

	BoardDetailResponse getBoard(Long userId, Long boardId);

	Page<BoardBaseResponse> getBoardList(Long userId, BoardSearchCondition boardSearchCondition, Pageable pageable);
}