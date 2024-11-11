package com.chatbot.backend.domain.remark.controller;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.chatbot.backend.domain.remark.dto.request.RemarkCreateRequestDto;
import com.chatbot.backend.domain.remark.dto.request.RemarkSearchCondition;
import com.chatbot.backend.domain.remark.dto.request.RemarkUpdateRequestDto;
import com.chatbot.backend.domain.remark.dto.response.RemarkBaseResponseDto;
import com.chatbot.backend.domain.remark.dto.response.RemarkDetailResponseDto;
import com.chatbot.backend.domain.remark.service.RemarkService;
import com.chatbot.backend.global.security.CustomUserDetails;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Tag(name = "Remark", description = "규정 게시판 비고 관련 API")
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/boards/remarks")
@Slf4j
public class RemarkController {
	private final RemarkService remarkService;

	@Operation(
		summary = "비고 생성",
		description = "새로운 비고를 생성합니다. 내용, 게시판 ID을 입력받습니다."
	)
	@PostMapping
	public ResponseEntity<RemarkDetailResponseDto> createRemark(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@Valid @RequestBody RemarkCreateRequestDto remarkCreateRequestDto
	) {
		return ResponseEntity.status(HttpStatus.CREATED)
			.body(remarkService.createRemark(userDetails.getId(), remarkCreateRequestDto));
	}

	@Operation(
		summary = "비고 수정",
		description = "비고를 수정합니다. 내용을 입력받습니다."
	)
	@PutMapping("/{remarkId}")
	public ResponseEntity<RemarkDetailResponseDto> updateRemark(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long remarkId,
		@Valid @RequestBody RemarkUpdateRequestDto updateRequestDto
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(remarkService.updateRemark(userDetails.getId(), remarkId, updateRequestDto));
	}

	@Operation(
		summary = "비고 삭제",
		description = "비고를 삭제합니다."
	)
	@DeleteMapping("/{remarkId}")
	public ResponseEntity<Void> deleteRemark(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long remarkId
	) {
		remarkService.deleteRemark(userDetails.getId(), remarkId);
		return ResponseEntity.status(HttpStatus.NO_CONTENT).build();
	}

	@Operation(
		summary = "비고 상세 조회",
		description = "비고를 상세 조회합니다."
	)
	@GetMapping("/{remarkId}")
	public ResponseEntity<RemarkDetailResponseDto> getRemark(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long remarkId
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(remarkService.getRemark(userDetails.getId(), remarkId));
	}

	@Operation(
		summary = "비고 목록 조회",
		description = "비고를 목록을 조회합니다."
	)
	@GetMapping
	public ResponseEntity<Page<RemarkBaseResponseDto>> getRemarkList(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@ModelAttribute RemarkSearchCondition remarkSearchCondition,
		@PageableDefault(size = 10, sort = "createdAt", direction = Sort.Direction.DESC) Pageable pageable
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(remarkService.getRemarkList(userDetails.getId(), remarkSearchCondition, pageable));
	}
}
