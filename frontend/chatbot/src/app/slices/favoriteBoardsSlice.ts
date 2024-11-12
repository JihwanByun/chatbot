import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import defaultAxios from "@apis/common";
import { BoardResponse } from "@apis/board/boardApi";

interface FavoriteBoardsState {
  favorites: BoardResponse[];
  loading: boolean;
  error: string | null;
  hasMore: boolean;
  currentPage: number;
}

const initialState: FavoriteBoardsState = {
  favorites: [],
  loading: false,
  error: null,
  hasMore: true,
  currentPage: 0,
};

export const fetchFavoriteBoards = createAsyncThunk(
  "favoriteBoards/fetchFavoriteBoards",
  async (page: number) => {
    const response = await defaultAxios.get("/api/boards/favorite", {
      params: {
        page,
        size: 10,
        sort: "createdAt,desc",
      },
    });
    return response.data;
  }
);

const favoriteBoardsSlice = createSlice({
  name: "favoriteBoards",
  initialState,
  reducers: {
    resetFavorites: (state) => {
      state.favorites = [];
      state.currentPage = 0;
      state.hasMore = true;
    },
    updateFavoriteStatus: (state, action) => {
      const { boardId, isFavorite, boardData } = action.payload;
      if (!isFavorite) {
        // 즐겨찾기 해제 시 목록에서 제거
        state.favorites = state.favorites.filter(
          (board) => board.boardId !== boardId
        );
      } else if (boardData) {
        // 즐겨찾기 추가 시 목록 맨 앞에 추가
        state.favorites = [boardData, ...state.favorites];
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFavoriteBoards.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchFavoriteBoards.fulfilled, (state, action) => {
        state.loading = false;
        if (action.payload.number === 0) {
          // 첫 페이지인 경우 기존 데이터 교체
          state.favorites = action.payload.content;
        } else {
          // 이후 페이지는 데이터 추가
          state.favorites = [...state.favorites, ...action.payload.content];
        }
        state.hasMore = !action.payload.last;
        state.currentPage = action.payload.number;
      })
      .addCase(fetchFavoriteBoards.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? "Failed to fetch favorite boards";
      });
  },
});

export const { resetFavorites, updateFavoriteStatus } =
  favoriteBoardsSlice.actions;
export default favoriteBoardsSlice.reducer;
