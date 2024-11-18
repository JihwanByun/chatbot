package com.chatbot.backend.global.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsUtils;

import com.chatbot.backend.global.jwt.ExceptionHandlerFilter;
import com.chatbot.backend.global.jwt.JwtFilter;
import com.chatbot.backend.global.jwt.JwtProvider;
import com.chatbot.backend.global.jwt.Role;

import lombok.RequiredArgsConstructor;

// Spring Security 설정을 담당
@RequiredArgsConstructor
@Configuration
@EnableWebSecurity    // Spring Security 활성화
@EnableMethodSecurity
public class SecurityConfig {
	private final JwtProvider jwtProvider;

	// 비밀번호 암호화
	@Bean
	public PasswordEncoder getPasswordEncoder() {
		return new BCryptPasswordEncoder();
	}

	// Security 필터 체인 구성
	@Bean
	public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
		http
			.cors(corsConfig ->
				corsConfig.disable())    // CORS 설정 비활성화 (CorsConfig에서 처리)
			.csrf(csrfConfig ->
				csrfConfig.disable())    // CSRF 보호 비활성화
			.formLogin(formLoginConfig ->
				formLoginConfig.disable())    // 폼 로그인 비활성화
			.httpBasic(httpBasicConfig ->
				httpBasicConfig.disable())    // HTTP Basic 인증 비활성화
			.sessionManagement(sessionManagement ->
				sessionManagement.sessionCreationPolicy(SessionCreationPolicy.STATELESS))    // 세션 사용하지 않음
			.authorizeHttpRequests(authorizeRequests ->
				authorizeRequests
					// URL 별 접근 권한 설정
					.requestMatchers(HttpMethod.OPTIONS, "/**")
					.permitAll()
					.requestMatchers(CorsUtils::isCorsRequest)
					.permitAll()
					.requestMatchers("/v3/api-docs/**", "/swagger-ui/**")
					.permitAll()    // Swagger 문서 접근 허용

					// 미인증 사용자 접근 허용
					.requestMatchers("/api/", "/api/users/**", "/api/files", "/api/notifications/**")
					.permitAll()

					// ADMIN 권한 설정
					.requestMatchers("/api/admin/**")
					.hasRole(Role.ADMIN.getKey())

					// ADMIN과 USER 모두 접근 가능
					// 권한에 따른 board 접근 제어
					.requestMatchers(HttpMethod.GET, "/api/boards/**", "/api/categories", "/api/files/**",
						"/api/feedbacks")
					.hasAnyAuthority(Role.ADMIN.getKey(), Role.USER.getKey(), Role.USER_WRITE.getKey())

					.requestMatchers(HttpMethod.POST, "/api/boards/{boardId}/favorite")
					.hasAnyAuthority(Role.USER.getKey(), Role.USER_WRITE.getKey(), Role.ADMIN.getKey())

					.requestMatchers(HttpMethod.DELETE, "/api/boards/{boardId}/favorite")
					.hasAnyAuthority(Role.USER.getKey(), Role.USER_WRITE.getKey(), Role.ADMIN.getKey())

					.requestMatchers(HttpMethod.POST, "/api/boards/**")
					.hasAuthority(Role.USER_WRITE.getKey())

					.requestMatchers(HttpMethod.PUT, "/api/boards/**")
					.hasAuthority(Role.USER_WRITE.getKey())

					.requestMatchers(HttpMethod.DELETE, "/api/boards/**")
					.hasAuthority(Role.USER_WRITE.getKey())

					// 권한에 따른 ADMIN 접근 제어
					.requestMatchers(HttpMethod.POST, "/api/categories/**")
					.hasAuthority(Role.ADMIN.getKey())

					.requestMatchers(HttpMethod.DELETE, "/api/categories/**")
					.hasAuthority(Role.ADMIN.getKey())

					// USER 권한 설정
					.requestMatchers("/api/chats/**", "/api/chatrooms/**")
					.hasAnyAuthority(Role.USER.getKey(), Role.USER_WRITE.getKey())

					// 그 외 모든 요청은 인증 필요
					.anyRequest()
					.authenticated()
			)
			// JWT 필터 추가
			.addFilterBefore(new JwtFilter(jwtProvider), UsernamePasswordAuthenticationFilter.class)
			.addFilterBefore(new ExceptionHandlerFilter(), JwtFilter.class);

		return http.build();
	}
}