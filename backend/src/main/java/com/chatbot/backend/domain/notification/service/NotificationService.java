package com.chatbot.backend.domain.notification.service;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.chatbot.backend.domain.notification.domain.NotificationEvent;
import com.chatbot.backend.domain.notification.event.Event;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

// NotificationService.java
@Service
@Slf4j
@RequiredArgsConstructor
public class NotificationService {
	private final Map<String, SseEmitter> emitters = new ConcurrentHashMap<>();

	private final Long TIME_OUT = 3600000L;

	public SseEmitter subscribe(String userId) {
		SseEmitter emitter = new SseEmitter(TIME_OUT);

		log.info("Subscribing to user {}", userId);

		try {
			// 연결 시 초기 이벤트 전송
			emitter.send(SseEmitter.event()
				.name("connect")
				.data("Connected!"));
			log.info("Finished subscribing to user {}", userId);
		} catch (IOException e) {
			log.error("Error sending initial events", e);
			emitter.complete();
			return emitter;
		}

		emitter.onCompletion(() -> {
			log.info("Emitter completed for user: {}", userId);
			emitters.remove(userId);
		});

		emitter.onTimeout(() -> {
			log.info("Emitter timeout for user: {}", userId);
			emitters.remove(userId);
		});

		emitters.put(userId, emitter);

		log.info("Emitter : {}", emitter);
		return emitter;
	}

	public void notify(String userId, Event event, String message, Map<String, Object> data) {
		SseEmitter emitter = emitters.get(userId);
		if (emitter != null) {
			try {
				emitter.send(SseEmitter.event()
					.name(event.getName())
					.data(NotificationEvent.of(event, message, data)));
			} catch (IOException e) {
				log.error("Error sending event: {} to user: {}", event, userId, e);
				emitters.remove(userId);
			}
		}
	}

	public void notifyAll(Event event, String message, Map<String, Object> data) {
		emitters.forEach((userId, emitter) -> {
			try {
				emitter.send(SseEmitter.event()
					.name(event.getName())
					.data(NotificationEvent.of(event, message, data)));
			} catch (IOException e) {
				log.error("Error broadcasting event: {} to user: {}", event, userId, e);
				emitters.remove(userId);
			}
		});
	}
}