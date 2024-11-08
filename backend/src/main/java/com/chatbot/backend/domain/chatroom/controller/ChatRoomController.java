package com.chatbot.backend.domain.chatroom.controller;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.chatbot.backend.domain.chatroom.dto.request.CreateChatRoomRequestDto;
import com.chatbot.backend.domain.chatroom.dto.response.CreateChatRoomResponseDto;
import com.chatbot.backend.domain.chatroom.dto.response.DeleteChatRoomResponseDto;
import com.chatbot.backend.domain.chatroom.dto.response.FindChatRoomDetailResponseDto;
import com.chatbot.backend.domain.chatroom.dto.response.FindChatRoomListResponseDto;
import com.chatbot.backend.domain.chatroom.service.ChatRoomServiceImpl;
import com.chatbot.backend.global.security.CustomUserDetails;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("api/chatrooms")
@RequiredArgsConstructor
public class ChatRoomController {

	private final ChatRoomServiceImpl chatRoomService;

	@PostMapping("")
	public ResponseEntity<CreateChatRoomResponseDto> createChatRoom(
		@AuthenticationPrincipal CustomUserDetails customUserDetails,
		@RequestBody CreateChatRoomRequestDto createChatRoomRequestDto) {

		CreateChatRoomResponseDto response = chatRoomService.createChatRoom(
			customUserDetails.getId(),
			createChatRoomRequestDto);

		return ResponseEntity.ok().body(response);
	}

	@GetMapping("")
	public ResponseEntity<FindChatRoomListResponseDto> findChatRoomList(
		@AuthenticationPrincipal CustomUserDetails userDetails,
		@RequestParam(defaultValue = "0") int page,
		@RequestParam(defaultValue = "10") int size
	) {

		Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());

		FindChatRoomListResponseDto response = chatRoomService.findChatRoomList(userDetails.getId(), pageable);

		return ResponseEntity.ok().body(response);
	}

	@GetMapping("/{chatRoomId}")
	public ResponseEntity<FindChatRoomDetailResponseDto> findChatRoomDetail(
		@PathVariable Long chatRoomId
	) {
		return ResponseEntity.ok().body(chatRoomService.findChatRoomDetail(chatRoomId));
	}

	@DeleteMapping("/{chatRoomId}")
	public ResponseEntity<DeleteChatRoomResponseDto> removeChatRoom(
		@PathVariable Long chatRoomId
	) {
		chatRoomService.deleteChatRoom(chatRoomId);

		return ResponseEntity.ok().build();
	}
}