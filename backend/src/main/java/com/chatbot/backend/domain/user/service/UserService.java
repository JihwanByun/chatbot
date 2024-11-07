package com.chatbot.backend.domain.user.service;

import com.chatbot.backend.domain.user.dto.request.LoginRequestDto;
import com.chatbot.backend.domain.user.dto.request.SignupRequestDto;
import com.chatbot.backend.global.jwt.Role;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

public interface UserService {
	void signUp(SignupRequestDto signupRequestDto);
	Role login(LoginRequestDto loginRequestDto, HttpServletResponse response);
	void logout(HttpServletRequest request, HttpServletResponse response);
	boolean isDuplicate(String email);
	void reissue(HttpServletRequest request, HttpServletResponse response);
}
