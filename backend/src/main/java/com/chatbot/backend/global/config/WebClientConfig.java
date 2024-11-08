package com.chatbot.backend.global.config;

import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import lombok.extern.slf4j.Slf4j;
import reactor.netty.http.client.HttpClient;

@Slf4j
@Configuration
public class WebClientConfig {
	private static final int TIMEOUT_SECONDS = 300;
	private static final int CONNECT_TIMEOUT_MILLIS = 10000;

	@Value("${ai.base.url}")
	private String AI_BASE_URL;

	@Bean
	public WebClient webClient() {
		HttpClient httpClient = HttpClient.create()
			.option(ChannelOption.CONNECT_TIMEOUT_MILLIS, CONNECT_TIMEOUT_MILLIS)
			.responseTimeout(Duration.ofSeconds(TIMEOUT_SECONDS))
			.doOnConnected(conn -> conn
				.addHandlerLast(new ReadTimeoutHandler(TIMEOUT_SECONDS))
				.addHandlerLast(new WriteTimeoutHandler(TIMEOUT_SECONDS)));

		return WebClient.builder()
			.baseUrl(AI_BASE_URL)
			.clientConnector(new ReactorClientHttpConnector(httpClient))
			.codecs(configurer -> configurer
				.defaultCodecs()
				.maxInMemorySize(16 * 1024 * 1024))
			.build();
	}
}