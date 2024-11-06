package com.chatbot.backend.global.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.repository.configuration.EnableRedisRepositories;
import org.springframework.data.redis.serializer.StringRedisSerializer;

import lombok.extern.slf4j.Slf4j;

@Configuration
@EnableRedisRepositories
@Slf4j
public class RedisConfig {

	@Value("${spring.data.redis.host}")
	String REDIS_HOST;

	@Value("${spring.data.redis.port}")
	int REDIS_PORT;

	// RedisConnectionFactory 설정
	@Bean
	public RedisConnectionFactory redisConnectionFactory() {
		log.info("Connecting to Redis {}:{}", REDIS_HOST,REDIS_PORT);
		return new LettuceConnectionFactory(REDIS_HOST, REDIS_PORT); // 필요 시 호스트와 포트를 수정하세요.
	}

	// RedisTemplate 설정
	@Bean
	public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
		RedisTemplate<String, Object> template = new RedisTemplate<>();
		template.setConnectionFactory(connectionFactory);

		// Key와 Value를 문자열로 직렬화
		template.setKeySerializer(new StringRedisSerializer());
		template.setValueSerializer(new StringRedisSerializer());

		// Hash Key와 Hash Value도 문자열로 직렬화 (필요 시)
		template.setHashKeySerializer(new StringRedisSerializer());
		template.setHashValueSerializer(new StringRedisSerializer());

		template.afterPropertiesSet();
		return template;
	}
}
