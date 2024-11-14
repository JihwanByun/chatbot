import { configureStore } from "@reduxjs/toolkit";
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import memberSlice from "./slices/memberSlice";
import feedbackSlice from "./slices/feedbackSlice";
import userSlice from "./slices/userSlice";
import treeSlice from "./slices/treeSlice";
import favoriteBoardsReducer from "./slices/favoriteBoardsSlice";

const userPersistConfig = {
  key: 'root',
  storage,
  whitelist: ['role', 'name', 'employeeNum', 'department', 'level'] // 유지하고 싶은 상태만 선택
};

const persistedReducer = persistReducer(userPersistConfig, userSlice);

export const store = configureStore({
	reducer: {
		members: memberSlice,
		feedbacks: feedbackSlice,
		users: persistedReducer,
		tree: treeSlice,
		favoriteBoards: favoriteBoardsReducer,
	},
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
});

export const persistor = persistStore(store);


export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;
