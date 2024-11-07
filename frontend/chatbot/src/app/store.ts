import { configureStore } from "@reduxjs/toolkit";
import memberSlice from "./slices/memberSlice";
import feedbackSlice from "./slices/feedbackSlice";
import userSlice from "./slices/userSlice";

export const store = configureStore({
	reducer: { members: memberSlice, feedbacks: feedbackSlice, users: userSlice },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;
