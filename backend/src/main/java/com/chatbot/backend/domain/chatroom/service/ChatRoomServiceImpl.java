package com.chatbot.backend.domain.chatroom.service;

import java.util.List;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.chatbot.backend.domain.chat.dto.ChatDto;
import com.chatbot.backend.domain.chat.repository.MongoChatRepository;
import com.chatbot.backend.domain.chatroom.dto.ChatRoomDto;
import com.chatbot.backend.domain.chatroom.dto.request.CreateChatRoomRequestDto;
import com.chatbot.backend.domain.chatroom.dto.response.CreateChatRoomResponseDto;
import com.chatbot.backend.domain.chatroom.dto.response.FindChatRoomDetailResponseDto;
import com.chatbot.backend.domain.chatroom.dto.response.FindChatRoomListResponseDto;
import com.chatbot.backend.domain.chatroom.entity.ChatRoom;
import com.chatbot.backend.domain.chatroom.repository.ChatRoomRepository;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.repository.UserRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ChatRoomServiceImpl implements ChatRoomService {

	private final ChatRoomRepository chatRoomRepository;
	private final UserRepository userRepository;
	private final MongoChatRepository chatRepository;

	@Override
	@Transactional
	public CreateChatRoomResponseDto createChatRoom(
		Long userId,
		CreateChatRoomRequestDto createChatRoomRequestDto) {

		User user = userRepository.findUserById(userId);

		ChatRoom chatroom = chatRoomRepository.save(
			ChatRoom.builder()
				.user(user)
				.content(createChatRoomRequestDto.getContent())
				.build());

		return new CreateChatRoomResponseDto(chatroom.getId(), chatroom.getTitle());
	}

	@Override
	public FindChatRoomListResponseDto findChatRoomList(Long userId,
		Pageable pageable) {

		Page<ChatRoom> chatRooms = chatRoomRepository.findAllByUserIdAndIsDeletedFalseOrderByCreatedAtDesc(
			userId,
			pageable);

		List<ChatRoomDto> chatRoomDtos = chatRooms.getContent().stream()
			.map(chatRoom -> new ChatRoomDto(chatRoom.getId(), chatRoom.getTitle(), chatRoom.getCreatedAt()))
			.toList();

		Page<ChatRoomDto> chatRoomDtoPage = new PageImpl<>(chatRoomDtos, pageable, chatRooms.getTotalElements());

		boolean hasNext = chatRoomDtoPage.hasNext();

		return new FindChatRoomListResponseDto(chatRoomDtoPage, hasNext);
	}

	@Override
	public FindChatRoomDetailResponseDto findChatRoomDetail(Long chatRoomId) {

		ChatRoom chatRoom = chatRoomRepository.findChatRoomById(chatRoomId);

		List<ChatDto> chats = chatRepository.findAllByChatRoomIdOrderByCreatedAtDesc(chatRoomId)
			.stream().map(chat -> new ChatDto(
				chat.getChatId().toHexString(),
				chat.getMemory(),
				chat.getIsPositive()
			)).toList();

		return new FindChatRoomDetailResponseDto(
			chatRoomId,
			chatRoom.getCreatedAt(),
			chats
		);
	}

	@Override
	public void deleteChatRoom(Long chatRoomId) {

		ChatRoom chatRoom = chatRoomRepository.findChatRoomById(chatRoomId);

		chatRoom.setDeleted(true);

		chatRoomRepository.save(chatRoom);

	}
}
