package com.chatbot.backend.domain.remark.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.remark.dto.request.RemarkCreateRequestDto;
import com.chatbot.backend.domain.remark.dto.request.RemarkSearchCondition;
import com.chatbot.backend.domain.remark.dto.request.RemarkUpdateRequestDto;
import com.chatbot.backend.domain.remark.dto.response.RemarkBaseResponseDto;
import com.chatbot.backend.domain.remark.dto.response.RemarkDetailResponseDto;

public interface RemarkService {
	RemarkDetailResponseDto createRemark(Long userId, RemarkCreateRequestDto remarkCreateRequestDto);

	RemarkDetailResponseDto updateRemark(Long userId, Long remarkId, RemarkUpdateRequestDto remarkUpdateRequestDto);

	void deleteRemark(Long userId, Long remarkId);

	RemarkDetailResponseDto getRemark(Long userId, Long remarkId);

	Page<RemarkBaseResponseDto> getRemarkList(Long userId, RemarkSearchCondition remarkSearchCondition,
		Pageable pageable);
}
