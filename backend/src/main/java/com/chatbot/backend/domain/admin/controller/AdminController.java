package com.chatbot.backend.domain.admin.controller;

import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.chatbot.backend.domain.admin.dto.response.PageResponseDto;
import com.chatbot.backend.domain.admin.service.AdminService;
import com.chatbot.backend.domain.category.dto.response.CategoryFindResponseDto;
import com.chatbot.backend.domain.category.dto.response.CategoryItemDto;
import com.chatbot.backend.global.security.CustomUserDetails;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Tag(name = "Admin", description = "관리자 관련 API")
@RestController
@Slf4j
@RequiredArgsConstructor
@RequestMapping("/api/admins")
public class AdminController {

	private final AdminService adminService;

	@GetMapping("/feedbacks")
	public ResponseEntity<PageResponseDto> findCategoryByPage(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		Boolean isPositive,
		@PageableDefault(size = 10, sort = "createdAt", direction = Sort.Direction.DESC) Pageable pageable)
	{

		Long userId = userDetails.getId();
		PageResponseDto response = adminService.findFeedbackByPage(userId, pageable);
		return ResponseEntity.ok().body(response);
	}

	@PostMapping("/categories")
	public ResponseEntity<CategoryFindResponseDto> createCategory(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@RequestBody String name){
		Long userId = userDetails.getId();
		adminService.createCategory(userId, name);
		return ResponseEntity.ok().build();
	}

	@Operation(
		summary = "카테고리 수정",
		description = "카테고리를 수정합니다. 카테고리 제목을 수정할 수 있습니다."
	)
	@PutMapping("/{categoryId}")
	public ResponseEntity<CategoryItemDto> updateCategory(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long categoryId,
		@RequestBody String name
	) {
		return ResponseEntity.status(HttpStatus.OK)
			.body(adminService.updateCategory(userDetails.getId(), categoryId, name));
	}

	@Operation(
		summary = "카테고리 삭제",
		description = "카테고리를 삭제합니다."
	)
	@DeleteMapping("/{categoryId}")
	public ResponseEntity<Void> deleteCategory(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@PathVariable Long categoryId
	) {
		adminService.deleteCategory(userDetails.getId(), categoryId);
		return ResponseEntity.status(HttpStatus.NO_CONTENT).build();
	}
}
