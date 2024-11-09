package com.chatbot.backend.domain.remark.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.chatbot.backend.domain.remark.dto.request.RemarkSearchCondition;
import com.chatbot.backend.domain.remark.entity.Remark;

public interface RemarkQueryRepository {
	Page<Remark> findAllByConditions(Long userId, RemarkSearchCondition remarkSearchCondition, Pageable pageable);
}
