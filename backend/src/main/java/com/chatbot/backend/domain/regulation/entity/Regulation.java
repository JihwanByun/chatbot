package com.chatbot.backend.domain.regulation.entity;

import java.util.List;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import com.chatbot.backend.domain.regulation.exception.RegulationInvalidException;
import com.chatbot.backend.global.shared.BaseTimeDocument;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

// 규정 Document 클래스
// MongoDB에 저장될 규정 정보 관리
@Document(collection = "regulations")
@Getter
@ToString
public class Regulation extends BaseTimeDocument {
	@Id
	private String id;
	private String title;
	private Integer level;
	private Long boardId;
	private List<RegulationItem> itemList;

	@Builder
	public Regulation(Long boardId, String title, Integer level, List<RegulationItem> itemList) {
		validateItemList(itemList);
		this.title = title;
		this.boardId = boardId;
		this.level = level;
		this.itemList = itemList;
	}

	//== 비즈니스 로직 ==//
	// 규정 업데이트
	public void updateRegulation(String title, Integer level, List<RegulationItem> itemList) {
		validateItemList(itemList);
		this.title = title == null ? this.title : title;
		this.level = level == null ? this.level : level;
		this.itemList = itemList;
	}

	//== 검증 로직 ==//
	// 규정 항목 리스트의 유효성 검증
	private void validateItemList(List<RegulationItem> itemList) {
		if (itemList == null || itemList.isEmpty()) {
			throw new RegulationInvalidException();
		}
	}
}
