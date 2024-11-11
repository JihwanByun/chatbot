package com.chatbot.backend.domain.regulation.entity;

import java.util.ArrayList;
import java.util.List;

import lombok.Builder;
import lombok.Getter;

// 규정 항목 클래스
@Getter
public class RegulationItem {
	private String title;
	private String content;
	private List<RegulationItem> itemList;

	@Builder
	public RegulationItem(String title, String content, List<RegulationItem> itemList) {
		this.title = title;
		this.content = content;
		this.itemList = itemList != null ? itemList : new ArrayList<>();
	}
}