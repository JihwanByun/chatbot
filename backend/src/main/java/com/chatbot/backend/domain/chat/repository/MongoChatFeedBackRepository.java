package com.chatbot.backend.domain.chat.repository;

import org.bson.types.ObjectId;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import com.chatbot.backend.domain.chat.entity.ChatFeedBack;

@Repository
public interface MongoChatFeedBackRepository extends MongoRepository<ChatFeedBack, ObjectId> {
	Page<ChatFeedBack> findAllByOrderByCreatedAtDesc(Pageable pageable);
	// ChatFeedBack findChatFeedBackByChatId(ObjectId objectId);
}
