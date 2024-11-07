package com.chatbot.backend.domain.chat.repository;

import org.bson.types.ObjectId;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.chat.entity.ChatFeedBack;

import io.lettuce.core.dynamic.annotation.Param;

@Repository
public interface MongoChatFeedBackRepository extends MongoRepository<ChatFeedBack, ObjectId> {
	// 긍정적인 피드백 (negativeReason이 null인 경우)
	@Query("{ 'negativeReason': null }")
	Page<ChatFeedBack> findAllByPositiveFeedback(Pageable pageable);

	// 부정적인 피드백 (negativeReason이 null이 아닌 경우)
	@Query("{ 'negativeReason': { $ne: null } }")
	Page<ChatFeedBack> findAllByNegativeFeedback(Pageable pageable);
}
