package com.chatbot.backend.global.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

// CORS(Cross-Origin Resource Sharing) 설정을 담당
@Configuration
public class CorsConfig implements WebMvcConfigurer {
	// application.yml 에서 개발 환경의 Origin 주소를 가져옴
	@Value("${cors.origin.local}")
	private String localOrigin;

	// application.yml 에서 개발 환경의 Origin 주소를 가져옴
	@Value("${cors.origin.development}")
	private String developmentOrigin;

	// 운영 환경의 Origin 주소를 가져옴
	@Value("${cors.origin.production}")
	private String productionOrigin;

	// CORS 설정을 추가하는 메소드
	@Override
	public void addCorsMappings(CorsRegistry registry) {
		registry.addMapping("/**")    // 모든 경로에 대해 CORS 설정 적용
			.allowedMethods("OPTIONS", "HEAD", "GET", "POST", "PUT", "PATCH", "DELETE")    // 허용할 HTTP 메소드 지정
			.allowCredentials(true)    // 인증 정보 (쿠키, Authorization 헤더 등) 허용
			.allowedOrigins(    // 허용할 출처 지정
				developmentOrigin,
				productionOrigin
			)
			.allowedHeaders("*")    // 모든 헤더 허용
			.exposedHeaders("*");    // 모든 헤더를 클라이언트에 노출
	}
}