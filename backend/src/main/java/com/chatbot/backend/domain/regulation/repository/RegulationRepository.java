package com.chatbot.backend.domain.regulation.repository;

import java.util.Optional;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.regulation.entity.Regulation;

@Repository
public interface RegulationRepository extends MongoRepository<Regulation, String> {
	// 게시글 ID를 통해 규정을 조회
	Optional<Regulation> findByBoardId(Long boardId);

	// 게시글 ID를 통해 규정을 삭제
	void deleteByBoardId(Long boardId);
}
