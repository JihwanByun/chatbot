package com.chatbot.backend.domain.regulation.service;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.chatbot.backend.domain.board.repository.BoardRepository;
import com.chatbot.backend.domain.regulation.dto.request.RegulationItemRequestDto;
import com.chatbot.backend.domain.regulation.dto.request.RegulationRequestDto;
import com.chatbot.backend.domain.regulation.dto.response.RegulationResponseDto;
import com.chatbot.backend.domain.regulation.entity.Regulation;
import com.chatbot.backend.domain.regulation.repository.RegulationRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class RegulationServiceImpl implements RegulationService {
	private final RegulationRepository regulationRepository;
	private final BoardRepository boardRepository;

	/**
	 * 새로운 규정 생성
	 *
	 * @param boardId
	 * @param regulationRequestDto
	 * @return
	 */
	@Override
	public RegulationResponseDto createRegulation(Long boardId, Integer level,
		RegulationRequestDto regulationRequestDto) {
		// Board 존재 여부 확인
		boardRepository.existsByIdOrElseThrow(boardId);

		// 규정 생성 및 저장
		Regulation regulation = regulationRequestDto.toDocument(boardId, level);
		regulationRepository.save(regulation);
		return RegulationResponseDto.of(regulation);
	}

	/**
	 * 규정 수정
	 *
	 * @param boardId
	 * @param regulationRequestDto
	 * @return
	 */
	@Override
	public RegulationResponseDto updateRegulation(Long boardId, RegulationRequestDto regulationRequestDto) {
		// Board 존재 여부 확인
		boardRepository.existsByIdOrElseThrow(boardId);

		// 기존 규정 조회
		Regulation regulation = regulationRepository.findByBoardIdOrElseThrow(boardId);

		// 규정 업데이트 및 저장
		regulation.updateRegulation(regulationRequestDto.itemList().stream()
			.map(RegulationItemRequestDto::toDocument)
			.toList());
		regulationRepository.save(regulation);
		return RegulationResponseDto.of(regulation);
	}

	/**
	 * 규정 조회
	 *
	 * @param boardId
	 * @return
	 */
	@Override
	public RegulationResponseDto getRegulation(Long boardId) {
		return RegulationResponseDto.of(regulationRepository.findByBoardId(boardId).orElse(null));
	}

	/**
	 * 규정 삭제
	 *
	 * @param boardId
	 */
	@Override
	public void deleteRegulation(Long boardId) {
		regulationRepository.deleteByBoardId(boardId);
	}
}
