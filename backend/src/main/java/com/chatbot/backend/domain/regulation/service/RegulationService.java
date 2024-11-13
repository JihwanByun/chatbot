package com.chatbot.backend.domain.regulation.service;

import com.chatbot.backend.domain.regulation.dto.request.RegulationRequestDto;
import com.chatbot.backend.domain.regulation.dto.response.RegulationResponseDto;

public interface RegulationService {
	RegulationResponseDto createRegulation(Long boardId, String title, Integer level, RegulationRequestDto regulationRequestDto);

	RegulationResponseDto updateRegulation(Long boardId, String title, Integer level, RegulationRequestDto regulationRequestDto);

	RegulationResponseDto getRegulation(Long boardId);

	void deleteRegulation(Long boardId);
}
