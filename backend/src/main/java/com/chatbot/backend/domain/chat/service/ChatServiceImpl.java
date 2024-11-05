package com.chatbot.backend.domain.chat.service;

import org.bson.types.ObjectId;
import org.springframework.stereotype.Service;

import com.chatbot.backend.domain.chat.dto.ChatDto;
import com.chatbot.backend.domain.chat.dto.request.CreateChatFeedBackRequestDto;
import com.chatbot.backend.domain.chat.dto.request.CreateChatRequestDto;
import com.chatbot.backend.domain.chat.dto.response.CreateChatFeedBackResponseDto;
import com.chatbot.backend.domain.chat.dto.response.CreateChatResponseDto;
import com.chatbot.backend.domain.chat.entity.Chat;
import com.chatbot.backend.domain.chat.entity.ChatFeedBack;
import com.chatbot.backend.domain.chat.repository.MongoChatFeedBackRepository;
import com.chatbot.backend.domain.chat.repository.MongoChatRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
public class ChatServiceImpl implements ChatService {

	private final MongoChatRepository mongoChatRepository;
	private final MongoChatFeedBackRepository chatFeedBackRepository;

	@Override
	public CreateChatResponseDto createChat(CreateChatRequestDto createChatRequestDto) {

		Chat savedChat = mongoChatRepository.save(
			Chat.builder()
				.chatRoomId(createChatRequestDto.getChatRoomId())
				.question(createChatRequestDto.getQuestion())
				.answer(createChatRequestDto.getAnswer())
				.build()
		);

		return new CreateChatResponseDto(
			savedChat.getChatId().toHexString(),
			savedChat.getQuestion(),
			savedChat.getAnswer()
		);

	}

	@Override
	public CreateChatFeedBackResponseDto createChatFeedBack(String chatId,
		CreateChatFeedBackRequestDto createChatFeedBackRequestDto) {

		Chat chat = mongoChatRepository.findById(new ObjectId(chatId)).
			orElseThrow();

		chat.setIsPositive(createChatFeedBackRequestDto.isPositive());
		mongoChatRepository.save(chat);

		ChatFeedBack savedFeedBack = chatFeedBackRepository.save(
			ChatFeedBack.builder()
				.chat(chat)
				.negativeReason(createChatFeedBackRequestDto.getNegativeReason())
				.build()
		);

		ChatDto chatDto = new ChatDto(
			chatId,
			chat.getQuestion(),
			chat.getAnswer()
		);

		return new CreateChatFeedBackResponseDto(
			chatDto,
			savedFeedBack.getNegativeReason()
		);

	}
}
