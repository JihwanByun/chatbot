package com.chatbot.backend.domain.notification.domain;

import java.time.LocalDateTime;
import java.util.Map;

import com.chatbot.backend.domain.notification.event.Event;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class NotificationEvent {
    private Event event;
    private String message;
    private LocalDateTime timestamp;
    private Map<String, Object> data;  // 추가 데이터를 위한 필드
    
    public static NotificationEvent of(Event event, String message, Map<String, Object> data) {
        return new NotificationEvent(event, message, LocalDateTime.now(), data);
    }

    public static NotificationEvent of(Event event, String message) {
        return of(event, message, null);
    }
}
