package com.chatbot.backend.domain.board.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.board.dto.request.BoardCreateRequestDto;
import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.dto.request.BoardUpdateRequestDto;
import com.chatbot.backend.domain.board.dto.response.BoardBaseResponseDto;
import com.chatbot.backend.domain.board.dto.response.BoardDetailResponseDto;
import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.entity.BoardLike;
import com.chatbot.backend.domain.board.repository.BoardLikeRepository;
import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.board.util.BoardValidator;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.repository.CategoryRepository;
import com.chatbot.backend.domain.file.service.FileService;
import com.chatbot.backend.domain.file.service.FileSummaryService;
import com.chatbot.backend.domain.regulation.dto.response.RegulationResponseDto;
import com.chatbot.backend.domain.regulation.service.RegulationService;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.repository.UserRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = false)
public class BoardServiceImpl implements BoardService {
	private static final String BOARD_UPLOAD_DIR = "boards";

	private final BoardRepository boardRepository;
	private final UserRepository userRepository;
	private final FileService fileService;
	private final CategoryRepository categoryRepository;
	private final BoardLikeRepository boardLikeRepository;
	private final BoardValidator boardValidator;
	private final RegulationService regulationService;
	private final FileSummaryService fileSummaryService;

	/**
	 * 새로운 게시글 작성
	 *
	 * @param userId
	 * @param boardCreateRequestDto
	 * @param file
	 * @return
	 */

	@Override
	@Transactional
	public BoardDetailResponseDto createBoard(Long userId, BoardCreateRequestDto boardCreateRequestDto,
		MultipartFile file) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Category category = categoryRepository.findByNameOrElseThrow(boardCreateRequestDto.category());

		boardValidator.validateUserWriteAuthorized(user);

		// Board 생성 (비즈니스 로직)
		Board board = boardCreateRequestDto.toEntity(user, category);
		board = boardRepository.save(board);

		// 파일이 있는 경우 비동기로 요약 처리 시작
		if (boardCreateRequestDto.hasFile()) {
			fileSummaryService.processFileSummaryAsync(file, board.getId());
			String fileUrl = fileService.saveFile(file, BOARD_UPLOAD_DIR);
			board.uploadFile(fileUrl);
		}

		// Regulation 생성 (규정 정보가 있을 경우에만)
		RegulationResponseDto regulationResponseDto = null;
		if (boardCreateRequestDto.regulationRequest() != null) {
			regulationResponseDto = regulationService.createRegulation(board.getId(),
				boardCreateRequestDto.regulationRequest());
		}

		return BoardDetailResponseDto.of(board, regulationResponseDto);
	}

	/**
	 * 기존 게시글 수정
	 *
	 * @param userId                사용자 ID
	 * @param boardId               게시글 ID
	 * @param boardUpdateRequestDto 게시글 수정 요청 DTO
	 * @param file                  수정 파일 (선택)
	 * @return 수정된 게시글 상세 응답 DTO
	 */
	@Override
	public BoardDetailResponseDto updateBoard(Long userId, Long boardId, BoardUpdateRequestDto boardUpdateRequestDto,
		MultipartFile file) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Category category = categoryRepository.findByNameOrElseThrow(boardUpdateRequestDto.category());
		Board board = boardRepository.findByIdOrElseThrow(boardId);

		boardValidator.validateBoardExists(board);

		// Board 수정 (비즈니스 로직)
		board.updateBoard(boardUpdateRequestDto, category, null);

		// File 저장
		if (boardUpdateRequestDto.hasFile()) {
			fileSummaryService.processFileSummaryAsync(file, board.getId());
			String fileUrl = fileService.saveFile(file, BOARD_UPLOAD_DIR);
			board.uploadFile(fileUrl);
		}

		RegulationResponseDto regulationResponse = null;
		if (!boardUpdateRequestDto.hasFile()) {
			regulationResponse = regulationService.updateRegulation(boardId, boardUpdateRequestDto.regulationRequest());
		}

		BoardLike boardLike = boardLikeRepository.findByBoardIdAndUserId(boardId, userId).orElse(null);
		return BoardDetailResponseDto.of(board, boardLike, regulationResponse);
	}

	/**
	 * 게시글 삭제
	 *
	 * @param userId  사용자 ID
	 * @param boardId 게시글 ID
	 */
	@Override
	public void deleteBoard(Long userId, Long boardId) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Board board = boardRepository.findByIdOrElseThrow(boardId);

		boardValidator.validateBoardExists(board);

		board.deleteBoard();
	}

	/**
	 * 게시글 상세 조회
	 *
	 * @param userId  사용자 ID
	 * @param boardId 게시글 ID
	 * @return 게시글 상세 응답 DTO
	 */
	@Override
	@Transactional(readOnly = true)
	public BoardDetailResponseDto getBoard(Long userId, Long boardId) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Board board = boardRepository.findByIdOrElseThrow(boardId);

		boardValidator.validateBoardExists(board);
		boardValidator.validateBoardAccess(user, board);

		BoardLike boardLike = boardLikeRepository.findByBoardIdAndUserId(boardId, userId).orElse(null);
		RegulationResponseDto regulationResponse = regulationService.getRegulation(boardId);

		return BoardDetailResponseDto.of(board, boardLike, regulationResponse);
	}

	/**
	 * 게시글 목록 조회
	 *
	 * @param userId               사용자 ID
	 * @param boardSearchCondition 검색 조건
	 * @param pageable             페이징 정보
	 * @return 게시글 리스트 응답 페이지
	 */
	@Override
	@Transactional(readOnly = true)
	public Page<BoardBaseResponseDto> getBoardList(Long userId, BoardSearchCondition boardSearchCondition,
		Pageable pageable) {
		return boardRepository.findAllByConditions(userId, boardSearchCondition, pageable)
			.map(BoardBaseResponseDto::of);
	}
}