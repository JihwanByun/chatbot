package com.chatbot.backend.global.jwt;

import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;

import com.chatbot.backend.global.jwt.exception.ExpiredTokenException;
import com.chatbot.backend.global.jwt.exception.InvalidTokenException;
import com.chatbot.backend.global.security.CustomUserDetails;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.JwtBuilder;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import lombok.extern.slf4j.Slf4j;

// JWT 생성 & 검증 & 인증에 관련된 기능
@Slf4j
@Component
public class JwtProvider {
	// JWT 권한을 표현
	private static final String AUTHORITIES_KEY = "auth";
	// JWT에서 사용자 ID를 저장하는 키
	private static final String USER_ID_KEY = "userId";
	// JWT 서명 알고리즘 (HS512 사용)
	private static final SignatureAlgorithm SIGNATURE_ALGORITHM = SignatureAlgorithm.HS512;

	@Value("${jwt.access_secret}")
	private String accessSecret;

	@Value("${jwt.refresh_secret}")
	private String refreshSecret;

	@Value("${jwt.access_token_expiration}")
	private long accessTokenExpiration;

	@Value("${jwt.refresh_token_expiration}")
	private long refreshTokenExpiration;

	/**
	 * 사용자 ID, 이메일, 권한 정보를 바탕으로 AccessToken 생성
	 * @param userId 사용자 ID
	 * @param email 사용자 Email
	 * @param role Role 사용자 권한
	 * @return 생성된 액세스 토큰
	 */
	public String createAccessToken(Long userId, String email, Role role) {
		return createToken(TokenType.ACCESS, userId, email, role);
	}

	/**
	 * 사용자 ID와 이메일을 바탕으로 RefreshToken을 생성
	 * @param userId 사용자 ID
	 * @param email 사용자 Email
	 * @return 생성된 리프레시 토큰
	 */
	public String createRefreshToken(Long userId, String email) {
		return createToken(TokenType.REFRESH, userId, email, null);
	}

	/**
	 * JWT 토큰을 생성
	 * @param tokenType 토큰 종류 (ACCESS or REFRESH)
	 * @param userId 사용자 ID
	 * @param email 사용자 Email
	 * @param role Role 사용자 권한 (액세스 토큰에만 포함)
	 * @return 생성된 JWT 토큰 문자열
	 */
	private String createToken(TokenType tokenType, Long userId, String email, Role role) {
		Date now = new Date();
		Date expiryDate = new Date(now.getTime() +
			(tokenType == TokenType.ACCESS ? accessTokenExpiration : refreshTokenExpiration));

		JwtBuilder jwtBuilder = Jwts.builder()
			.setSubject(email)
			.claim(USER_ID_KEY, userId)
			.setIssuedAt(now)
			.setExpiration(expiryDate)
			.signWith(SIGNATURE_ALGORITHM,
				tokenType == TokenType.ACCESS ? accessSecret : refreshSecret);

		if (role != null && tokenType == TokenType.ACCESS) {
			jwtBuilder.claim(AUTHORITIES_KEY, role.getKey());
		}

		return jwtBuilder.compact();
	}

	/**
	 * 주어진 토큰으로부터 인증 정보를 추출
	 * @param token JWT 토큰
	 * @return 추출된 Authentication 객체
	 */
	public Authentication getAuthentication(String token) {
		Claims claims = parseClaims(token, false);

		Collection<? extends GrantedAuthority> authorities = extractAuthorities(claims);
		CustomUserDetails userDetails = extractUserDetails(claims);

		return new UsernamePasswordAuthenticationToken(userDetails, "", authorities);
	}

	/**
	 * 클레임에서 권한 정보를 추출
	 * @param claims JWT 클레임
	 * @return 권한 목록
	 */
	private Collection<? extends GrantedAuthority> extractAuthorities(Claims claims) {
		return Arrays.stream(claims.get(AUTHORITIES_KEY).toString().split(","))
			.map(SimpleGrantedAuthority::new)
			.collect(Collectors.toList());
	}

	/**
	 * 클레임에서 사용자 정보를 추출
	 * @param claims JWT 클레임
	 * @return 사용자 정보가 담긴 CustomUserDetails 객체
	 */
	private CustomUserDetails extractUserDetails(Claims claims) {
		Long userId = ((Number)claims.get(USER_ID_KEY)).longValue();
		String email = claims.getSubject();
		Role role = Role.valueOf(claims.get(AUTHORITIES_KEY).toString());

		return CustomUserDetails.of(userId, email, role);
	}

	/**
	 * JWT 토큰의 유효성 검증
	 * @param token 검증할 JWT 토큰
	 * @param isRefreshToken Refresh Token 여부
	 * @throws ExpiredTokenException Token이 만료된 경우 예외
	 * @throws InvalidTokenException Token이 유효하지 않은 경우 예외
	 */
	public void validateToken(String token, boolean isRefreshToken) {
		try {
			parseClaims(token, isRefreshToken);
		} catch (ExpiredJwtException e) {
			throw new ExpiredTokenException();
		} catch (JwtException | IllegalArgumentException e) {
			throw new InvalidTokenException();
		}
	}

	/**
	 * JWT 토큰에서 클레임을 파싱하는 메서드
	 * @param token 파싱할 JWT 토큰
	 * @param isRefreshToken Refresh Token 여부
	 * @return JWT 클레임
	 * @throws ExpiredTokenException Token이 만료된 경우 예외 발생
	 */
	public Claims parseClaims(String token, boolean isRefreshToken) {
		return Jwts.parser()
			.setSigningKey(isRefreshToken ? refreshSecret : accessSecret)
			.parseClaimsJws(token)
			.getBody();
	}

	private enum TokenType {
		ACCESS, REFRESH
	}
}
