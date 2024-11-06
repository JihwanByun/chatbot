package com.chatbot.backend.domain.board.service;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.BDDMockito.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.board.dto.request.BoardCreateRequestDto;
import com.chatbot.backend.domain.board.dto.response.BoardDetailResponseDto;
import com.chatbot.backend.domain.board.entity.Board;
import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.category.entity.Category;
import com.chatbot.backend.domain.category.repository.CategoryRepository;
import com.chatbot.backend.domain.file.service.FileService;
import com.chatbot.backend.domain.user.entity.User;
import com.chatbot.backend.domain.user.repository.UserRepository;

@ExtendWith(MockitoExtension.class)
public class BoardServiceImplTest {

	@InjectMocks
	private BoardServiceImpl boardService;

	@Mock
	private BoardRepository boardRepository;

	@Mock
	private UserRepository userRepository;

	@Mock
	private CategoryRepository categoryRepository;

	@Mock
	private FileService fileService;

	@Test
	@DisplayName("게시글 생성 성공 테스트")
	void createBoard_Success() {
		// Given
		Long userId = 1L;
		BoardCreateRequestDto request = new BoardCreateRequestDto(
			"테스트 제목",
			"테스트 내용",
			"카테고리1",
			1
		);

		MultipartFile file = new MockMultipartFile(
			"file",
			"test.txt",
			"text/plain",
			"Hello, World!".getBytes()
		);

		User user = User.builder()
			.email("test@example.com")
			.name("테스트유저")
			.profileUrl("http://example.com/profile.jpg")
			.build();

		Category category = Category.builder()
			.name("카테고리1")
			.build();

		String fileUrl = "http://example.com/test.txt";

		Board savedBoard = Board.builder()
			.title(request.title())
			.contents(request.contents())
			.level(request.level())
			.fileUrl(fileUrl)
			.user(user)
			.category(category)
			.build();

		given(userRepository.findByIdOrElseThrow(userId)).willReturn(user);
		given(categoryRepository.findByNameOrElseThrow(request.category())).willReturn(category);
		given(fileService.saveFile(any(), any())).willReturn(fileUrl);
		given(boardRepository.save(any(Board.class))).willReturn(savedBoard);

		// When
		BoardDetailResponseDto response = boardService.createBoard(userId, request, file);

		// Then
		assertThat(response).isNotNull();
		assertThat(response.boardId()).isEqualTo(savedBoard.getId());
		assertThat(response.title()).isEqualTo(request.title());
		assertThat(response.contents()).isEqualTo(request.contents());
		assertThat(response.level()).isEqualTo(request.level());
		assertThat(response.category()).isEqualTo(category.getName());
		assertThat(response.fileUrl()).isEqualTo(fileUrl);
		assertThat(response.writer().email()).isEqualTo(user.getEmail());
		assertThat(response.writer().name()).isEqualTo(user.getName());
		assertThat(response.writer().profileUrl()).isEqualTo(user.getProfileUrl());

		verify(userRepository).findByIdOrElseThrow(userId);
		verify(categoryRepository).findByNameOrElseThrow(request.category());
		verify(fileService).saveFile(any(), any());
		verify(boardRepository).save(any(Board.class));
	}

	@Test
	@DisplayName("파일 없이 게시글 생성 성공 테스트")
	void createBoard_WithoutFile_Success() {
		// Given
		Long userId = 1L;
		BoardCreateRequestDto request = new BoardCreateRequestDto(
			"테스트 제목",
			"테스트 내용",
			"카테고리1",
			1
		);

		User user = User.builder()
			.email("test@example.com")
			.name("테스트유저")
			.profileUrl("http://example.com/profile.jpg")
			.build();

		Category category = Category.builder()
			.name("카테고리1")
			.build();

		Board savedBoard = Board.builder()
			.title(request.title())
			.contents(request.contents())
			.level(request.level())
			.user(user)
			.category(category)
			.build();

		given(userRepository.findByIdOrElseThrow(userId)).willReturn(user);
		given(categoryRepository.findByNameOrElseThrow(request.category())).willReturn(category);
		given(boardRepository.save(any(Board.class))).willReturn(savedBoard);

		// When
		BoardDetailResponseDto response = boardService.createBoard(userId, request, null);

		// Then
		assertThat(response).isNotNull();
		assertThat(response.boardId()).isEqualTo(savedBoard.getId());
		assertThat(response.fileUrl()).isNull();

		verify(userRepository).findByIdOrElseThrow(userId);
		verify(categoryRepository).findByNameOrElseThrow(request.category());
		verify(boardRepository).save(any(Board.class));
	}
}