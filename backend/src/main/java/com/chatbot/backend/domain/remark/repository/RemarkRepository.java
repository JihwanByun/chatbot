package com.chatbot.backend.domain.remark.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.remark.entity.Remark;
import com.chatbot.backend.domain.remark.exception.RemarkNotFoundException;

@Repository
public interface RemarkRepository extends JpaRepository<Remark, Long>, RemarkQueryRepository {
	default Remark findByIdOrElseThrow(Long id) {
		return findById(id).orElseThrow(RemarkNotFoundException::new);
	}
}
