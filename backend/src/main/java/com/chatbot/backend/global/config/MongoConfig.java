package com.chatbot.backend.global.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.config.EnableMongoAuditing;

// MongoDB 설정 클래스
@Configuration
@EnableMongoAuditing // MongoDB Auditing 활성화 (생성 및 수정 시간 자동 관리)
public class MongoConfig {
}
