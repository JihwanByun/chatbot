// Redux Toolkit을 사용하여 member 상태 관리하는 slice를 정의

import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { findMemberListByPage } from "@apis/chat/adminApi"; // API 함수 가져오기

// members 관련 상태를 정의하는 인터페이스
interface MemberState {
	members: Member;
	loading: boolean;
	error: string | null;
}

// 각 멤버의 속성을 정의하는 타입
type Contents = {
	userId: number;
	email: string;
	name: string;
	employeeNum: string;
	department: string;
	level: number;
	role: string;
};

// Member 전체 리스트의 구조를 정의하는 타입
type Member = {
	content: Contents[];
	page: number;
	size: number;
	totalPages: number;
	totalElements: number;
};

// 초기 상태 정의
const initialState: MemberState = {
	members: {
		content: [],
		page: 0,
		size: 0,
		totalPages: 0,
		totalElements: 0,
	},
	loading: false,
	error: null,
};

// 비동기 Thunk: 페이지별로 멤버 데이터를 가져오는 함수
export const fetchMembersByPage = createAsyncThunk(
	"members/fetchMembersByPage",
	async (params: {
		page: number;
		size: number;
		sortBy: string;
		sortDir: string;
	}) => {
		// API 호출 후 데이터 반환
		const response = await findMemberListByPage(
			params.page - 1,
			params.size,
			params.sortBy,
			params.sortDir
		);
		console.log(response.data)
		return response.data;
	}
);

// Redux slice 생성
const memberSlice = createSlice({
	name: "members",
	initialState,
	reducers: {}, // 현재 동기 reducer 함수는 필요하지 않아 빈 객체로 둠
	extraReducers: (builder) => {
		builder
			.addCase(fetchMembersByPage.pending, (state) => {
				// 요청 중 상태를 업데이트
				state.loading = true;
				state.error = null;
			})
			.addCase(
				fetchMembersByPage.fulfilled,
				(state, action: PayloadAction<Member>) => {
					// 요청 성공 시 상태를 업데이트
					console.log(action.payload)

					state.members = action.payload;
					// state.members.totalPages = action.payload.totalPages;
					state.loading = false;
				}
			)
			.addCase(fetchMembersByPage.rejected, (state, action) => {
				// 요청 실패 시 에러 메시지 설정
				state.loading = false;
				state.error = action.error.message || "Failed to fetch members";
			});
	},
});

export default memberSlice.reducer; // reducer를 기본으로 export
