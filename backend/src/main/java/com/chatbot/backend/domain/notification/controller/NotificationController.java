package com.chatbot.backend.domain.notification.controller;

import java.util.Map;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.chatbot.backend.domain.notification.event.Event;
import com.chatbot.backend.domain.notification.service.NotificationService;

import lombok.RequiredArgsConstructor;

// NotificationController.java
@RestController
@RequestMapping("/api/notifications")
@RequiredArgsConstructor
public class NotificationController {
    private final NotificationService notificationService;

    @GetMapping(value = "/subscribe/{userId}", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter subscribe(@PathVariable String userId) {
        return notificationService.subscribe(userId);
    }

    @PostMapping("/send/{userId}/{eventType}")
    public ResponseEntity<Void> sendNotification(
        @PathVariable String userId,
        @PathVariable Event eventType,
        @RequestBody Map<String, Object> eventData
    ) {
        notificationService.notify(userId, eventType,
            (String) eventData.get("message"), eventData);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/broadcast/{eventType}")
    public ResponseEntity<Void> broadcast(
        @PathVariable Event eventType,
        @RequestBody Map<String, Object> eventData
    ) {
        notificationService.notifyAll(eventType,
            (String) eventData.get("message"), eventData);
        return ResponseEntity.ok().build();
    }
}