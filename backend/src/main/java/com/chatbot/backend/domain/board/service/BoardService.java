package com.chatbot.backend.domain.board.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.board.dto.request.BoardCreateRequestDto;
import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.dto.request.BoardUpdateRequestDto;
import com.chatbot.backend.domain.board.dto.response.BoardBaseResponseDto;
import com.chatbot.backend.domain.board.dto.response.BoardDetailResponseDto;

public interface BoardService {
	BoardDetailResponseDto createBoard(Long userId, BoardCreateRequestDto boardCreateRequestDto,
		MultipartFile file);

	BoardDetailResponseDto updateBoard(Long userId, Long boardId, BoardUpdateRequestDto boardUpdateRequestDto,
		MultipartFile file);

	void deleteBoard(Long userId, Long boardId);

	BoardDetailResponseDto getBoard(Long userId, Long boardId);

	Page<BoardBaseResponseDto> getBoardList(Long userId, BoardSearchCondition boardSearchCondition, Pageable pageable);
}