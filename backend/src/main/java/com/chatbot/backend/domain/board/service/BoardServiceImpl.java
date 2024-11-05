package com.chatbot.backend.domain.board.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.board.dto.request.BoardCreateRequest;
import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.dto.request.BoardUpdateRequest;
import com.chatbot.backend.domain.board.dto.response.BoardBaseResponse;
import com.chatbot.backend.domain.board.dto.response.BoardDetailResponse;
import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.entity.BoardLike;
import com.chatbot.backend.domain.board.repository.BoardLikeRepository;
import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.board.util.BoardValidator;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.repository.CategoryRepository;
import com.chatbot.backend.domain.file.service.FileService;
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

	/**
	 * 새로운 게시글 작성
	 * @param userId 사용자 ID
	 * @param boardCreateRequest 게시글 생성 요청 DTO
	 * @param file 업로드 파일 (선택)
	 * @return 생성된 게시글 상세 응답 DTO
	 */
	@Override
	public BoardDetailResponse createBoard(Long userId, BoardCreateRequest boardCreateRequest, MultipartFile file) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Category category = categoryRepository.findByNameOrElseThrow(boardCreateRequest.category());

		boardValidator.validateUserWriteAuthorized(user);

		// Board 생성 (비즈니스 로직)
		Board board = boardCreateRequest.toEntity(user, category, fileService.saveFile(file, BOARD_UPLOAD_DIR));

		return BoardDetailResponse.of(boardRepository.save(board));
	}

	/**
	 * 기존 게시글 수정
	 * @param userId 사용자 ID
	 * @param boardId 게시글 ID
	 * @param boardUpdateRequest 게시글 수정 요청 DTO
	 * @param file 수정 파일 (선택)
	 * @return 수정된 게시글 상세 응답 DTO
	 */
	@Override
	public BoardDetailResponse updateBoard(Long userId, Long boardId, BoardUpdateRequest boardUpdateRequest,
		MultipartFile file) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Category category = categoryRepository.findByNameOrElseThrow(boardUpdateRequest.category());
		Board board = boardRepository.findByIdOrElseThrow(boardId);

		boardValidator.validateBoardExists(board);
		boardValidator.validateUserAuthorization(user, board);

		// Board 수정 (비즈니스 로직)
		board.updateBoard(boardUpdateRequest, category, fileService.saveFile(file, BOARD_UPLOAD_DIR));

		BoardLike boardLike = boardLikeRepository.findByBoardIdAndUserId(boardId, userId).orElse(null);
		return BoardDetailResponse.of(board, boardLike);
	}

	/**
	 * 게시글 삭제
	 * @param userId 사용자 ID
	 * @param boardId 게시글 ID
	 */
	@Override
	public void deleteBoard(Long userId, Long boardId) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Board board = boardRepository.findByIdOrElseThrow(boardId);

		boardValidator.validateBoardExists(board);
		boardValidator.validateUserAuthorization(user, board);

		board.deleteBoard();
	}

	/**
	 * 게시글 상세 조회
	 * @param userId 사용자 ID
	 * @param boardId 게시글 ID
	 * @return 게시글 상세 응답 DTO
	 */
	@Override
	@Transactional(readOnly = true)
	public BoardDetailResponse getBoard(Long userId, Long boardId) {
		// 검증 & 조회
		User user = userRepository.findByIdOrElseThrow(userId);
		Board board = boardRepository.findByIdOrElseThrow(boardId);

		boardValidator.validateBoardExists(board);
		boardValidator.validateBoardAccess(user, board);

		BoardLike boardLike = boardLikeRepository.findByBoardIdAndUserId(boardId, userId).orElse(null);
		return BoardDetailResponse.of(board, boardLike);
	}

	/**
	 * 게시글 목록 조회
	 * @param userId 사용자 ID
	 * @param boardSearchCondition 검색 조건
	 * @param pageable 페이징 정보
	 * @return 게시글 리스트 응답 페이지
	 */
	@Override
	@Transactional(readOnly = true)
	public Page<BoardBaseResponse> getBoardList(Long userId, BoardSearchCondition boardSearchCondition,
		Pageable pageable) {
		return boardRepository.findAllByConditions(userId, boardSearchCondition, pageable).map(BoardBaseResponse::of);
	}
}