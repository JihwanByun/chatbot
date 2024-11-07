// userSlice.ts
import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { LoginDTO } from "@pages/login/LoginDTO";
import { login as loginAPI } from "@apis/chat/userApi";
import axios from "axios";

interface UserState {
	role: string | null;
	loading: boolean;
	error: string | null;
}

const initialState: UserState = {
	role: null,
	loading: false,
	error: null,
};

// 비동기 thunk 함수로 로그인 처리
export const loginUser = createAsyncThunk(
	"user/loginUser",
	async (loginData: LoginDTO, { rejectWithValue }) => {
		try {
			// login API 호출
			const response = await loginAPI(loginData);
			console.log("Response Data:", response.data);
			console.log(response.headers.authorization);

			const { token, role } = response.data;

			// 인증 토큰 설정
			localStorage.setItem("Authorization", token);

			// role과 token 반환
			return { role };
		} catch (error) {
			if (axios.isAxiosError(error) && error.response) {
				return rejectWithValue(error.response.data);
			}
			return rejectWithValue({ message: "Unexpected error occurred" });
		}
	}
);

const userSlice = createSlice({
	name: "user",
	initialState,
	reducers: {
		logout(state) {
			state.role = null;
		},
	},
	extraReducers: (builder) => {
		builder
			.addCase(loginUser.pending, (state) => {
				state.loading = true;
				state.error = null;
			})
			.addCase(
				loginUser.fulfilled,
				(
					state,
					action: PayloadAction<{
						role: string;
					}>
				) => {
					console.log("Fulfilled Action Payload:", action.payload);
					state.loading = false;
					state.role = action.payload.role;
				}
			)
			.addCase(loginUser.rejected, (state, action: PayloadAction<any>) => {
				state.loading = false;
				state.error = action.payload?.message || "로그인 실패";
			});
	},
});

export const { logout } = userSlice.actions;

export default userSlice.reducer;
