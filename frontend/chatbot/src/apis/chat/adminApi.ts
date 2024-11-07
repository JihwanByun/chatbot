// axios를 사용해 API 요청을 수행하는 함수들

import { defaultAxios } from "@apis/common";

// 페이지별로 멤버 리스트를 가져오는 API 호출 함수
export const findMemberListByPage = async (
	page: number,
	size: number,
	sortBy: string,
	sortDir: string
) => {
	// API 요청 및 결과 반환
	const result = defaultAxios.get("/api/admins/users", {
		headers: {
			Authorization: localStorage.getItem("Authorization"),
		},
		params: {
			page,
			size,
			sortBy,
			sortDir,
		},
	});
	console.log("Member result!!! : ", result);
	return result;
};

// 페이지별로 피드백 리스트를 가져오는 API 호출 함수
export const findFeedbackListByPage = async (
	isPositive: boolean | null,
	page: number,
	size: number,
	sortBy: string,
	sortDir: string
) => {
	// API 요청 및 결과 반환
	const result = defaultAxios.get(`/api/admins/feedbacks`, {
		headers: {
			Authorization: localStorage.getItem("Authorization"),
		},
		params: {
			isPositive,
			page,
			size,
			sort: `${sortBy},${sortDir}`, // `sort` 파라미터를 한 문자열로 조합
		},
	});
	return result;
};
