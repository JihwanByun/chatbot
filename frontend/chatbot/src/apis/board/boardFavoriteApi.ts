import defaultAxios from "@apis/common";

/**
 * 게시글 즐겨찾기 생성 API
 */
export const createFavoriteAPI = async (boardId: number): Promise<boolean> => {
  try {
    const response = await defaultAxios.post(`/api/boards/${boardId}/favorite`);
    return response.status === 201;
  } catch (error) {
    console.error("Error in createFavoriteAPI:", error);
    return false;
  }
};

/**
 * 게시글 즐겨찾기 삭제 API
 */
export const deleteFavoriteAPI = async (boardId: number): Promise<boolean> => {
  try {
    const response = await defaultAxios.delete(
      `/api/boards/${boardId}/favorite`
    );
    return response.status === 204;
  } catch (error) {
    console.error("Error in deleteFavoriteAPI:", error);
    return false;
  }
};
