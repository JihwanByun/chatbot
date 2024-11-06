package com.chatbot.backend.domain.board.controller;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Slice;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.board.dto.request.BoardCreateRequest;
import com.chatbot.backend.domain.board.dto.request.BoardSearchCondition;
import com.chatbot.backend.domain.board.dto.request.BoardUpdateRequest;
import com.chatbot.backend.domain.board.dto.response.BoardBaseResponse;
import com.chatbot.backend.domain.board.dto.response.BoardDetailResponse;
import com.chatbot.backend.domain.board.service.BoardLikeService;
import com.chatbot.backend.domain.board.service.BoardService;
import com.chatbot.backend.global.security.CustomUserDetails;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Tag(name = "Board", description = "규정 게시판 관련 API")
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/boards")
@Slf4j
public class BoardController {
	private final BoardService boardService;
	private final BoardLikeService boardLikeService;

	@Operation(
		summary = "게시글 생성",
		description = "새로운 게시글을 생성합니다. 제목, 내용, 카테고리, 레벨 정보와 선택적으로 파일을 첨부할 수 있습니다."
	)
	@PostMapping(consumes = {MediaType.APPLICATION_JSON_VALUE, MediaType.MULTIPART_FORM_DATA_VALUE})
	public ResponseEntity<BoardDetailResponse> createBoard(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@Valid @RequestPart(value = "request") BoardCreateRequest boardCreateRequest,
		@RequestPart(value = "file", required = false) MultipartFile file) {
		return ResponseEntity.status(HttpStatus.CREATED)
			.body(boardService.createBoard(userDetails.getId(), boardCreateRequest, file));
	}

	@Operation(
		summary = "게시글 수정",
		description = "게시글을 수정합니다. 제목, 내용, 카테고리, 레벨 정보와 선택적으로 파일을 첨부할 수 있습니다."
	)
	@PutMapping(value = "/{boardId}", consumes = {MediaType.APPLICATION_JSON_VALUE,
		MediaType.MULTIPART_FORM_DATA_VALUE})
	public ResponseEntity<BoardDetailResponse> updateBoard(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long boardId,
		@Valid @RequestPart(value = "request") BoardUpdateRequest boardUpdateRequest,
		@RequestPart(value = "file", required = false) MultipartFile file) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(boardService.updateBoard(userDetails.getId(), boardId, boardUpdateRequest, file));
	}

	@Operation(
		summary = "게시글 삭제",
		description = "게시글을 삭제합니다."
	)
	@DeleteMapping("/{boardId}")
	public ResponseEntity<Void> deleteBoard(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long boardId
	) {
		boardService.deleteBoard(userDetails.getId(), boardId);
		return ResponseEntity.status(HttpStatus.NO_CONTENT).build();
	}

	@Operation(
		summary = "게시글 상세 조회",
		description = "게시글을 상세 조회합니다.  게시글 제목, 내용, 카테고리, 레벨 정보와 선택적으로 파일, 작성자의 정보를 가져올 수 있습니다."
	)
	@GetMapping("/{boardId}")
	public ResponseEntity<BoardDetailResponse> getBoard(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long boardId
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(boardService.getBoard(userDetails.getId(), boardId));
	}

	@Operation(
		summary = "게시글 즐겨찾기 생성",
		description = "게시글에 즐겨찾기를 추가합니다."
	)
	@PostMapping("/{boardId}/favorite")
	public ResponseEntity<Void> createBoardLike(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long boardId
	) {
		boardLikeService.createBoardLike(userDetails.getId(), boardId);
		return ResponseEntity.status(HttpStatus.CREATED).build();
	}

	@Operation(
		summary = "게시글 즐겨찾기 취소",
		description = "게시글에 추가된 즐겨찾기를 취소합니다."
	)
	@DeleteMapping("/{boardId}/favorite")
	public ResponseEntity<Void> deleteBoardLike(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long boardId
	) {
		boardLikeService.deleteBoardLike(userDetails.getId(), boardId);
		return ResponseEntity.status(HttpStatus.NO_CONTENT).build();
	}

	@Operation(
		summary = "게시글 목록 조회",
		description = "게시글 목록을 조회합니다."
	)
	@GetMapping
	public ResponseEntity<Page<BoardBaseResponse>> getBoardList(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@ModelAttribute BoardSearchCondition boardSearchCondition,
		@PageableDefault(size = 10, sort = "createdAt", direction = Sort.Direction.DESC) Pageable pageable
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(boardService.getBoardList(userDetails.getId(), boardSearchCondition, pageable));
	}

	@Operation(
		summary = "게시글 즐겨찾기 목록 조회",
		description = "게시글에 즐겨찾기 목록을 조회합니다."
	)
	@GetMapping("/favorite")
	public ResponseEntity<Slice<BoardBaseResponse>> getFavoriteBoardList(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PageableDefault(size = 10, sort = "createdAt", direction = Sort.Direction.DESC) Pageable pageable
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(boardLikeService.getFavoriteBoardList(userDetails.getId(), pageable));
	}
}