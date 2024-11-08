import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { findFeedbackListByPage } from "@apis/chat/adminApi"; // Ensure this path matches your project structure

interface FeedbackState {
	feedbacks: Feedback;
	loading: boolean;
	error: string | null;
}

type Contents = {
	chatId: string;
	question: string;
	answer: string;
	isPositive: boolean;
	negativeReason: string;
	createdAt: Date;
};

type Feedback = {
	contents: Contents[];
	page: number;
	size: number;
	totalPages: number;
	totalElements: number;
};

const initialState: FeedbackState = {
	feedbacks: {
		contents: [],
		page: 0,
		size: 0,
		totalPages: 0,
		totalElements: 0,
	},
	loading: false,
	error: null,
};

export const fetchFeedbacksByPage = createAsyncThunk(
	"feedbacks/fetchFeedbacksByPage",
	async (params: {
		isPositive: boolean | null;
		page: number;
		size: number;
		sortBy: string;
		sortDir: string;
	}) => {
		const response = await findFeedbackListByPage(
			params.isPositive,
			params.page - 1,
			params.size,
			params.sortBy,
			params.sortDir
		);
		return response.data;
	}
);

const feedbackSlice = createSlice({
	name: "feedback",
	initialState,
	reducers: {},
	extraReducers: (builder) => {
		builder
			.addCase(fetchFeedbacksByPage.pending, (state) => {
				state.loading = true;
				state.error = null;
			})
			.addCase(
				fetchFeedbacksByPage.fulfilled,
				(state, action: PayloadAction<Feedback>) => {
					state.feedbacks = action.payload;
					state.loading = false;
				}
			)
			.addCase(fetchFeedbacksByPage.rejected, (state, action) => {
				state.loading = false;
				state.error = action.error.message || "Failed to fetch members";
			});
	},
});

export default feedbackSlice.reducer;
