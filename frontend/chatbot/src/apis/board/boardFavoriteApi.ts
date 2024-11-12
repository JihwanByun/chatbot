import defaultAxios from "@apis/common";

/**
 * 게시글 즐겨찾기 생성 API
 * @param boardId - 즐겨찾기할 게시글의 ID
 * @returns Promise<boolean> - 성공 여부 (201 상태코드인 경우 true)
 *
 * - POST 요청을 통해 특정 게시글을 즐겨찾기에 추가
 * - 성공 시 201(Created) 상태코드 반환
 * - 실패 시 false 반환 및 에러 로깅
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
 * @param boardId - 즐겨찾기 해제할 게시글의 ID
 * @returns Promise<boolean> - 성공 여부 (204 상태코드인 경우 true)
 *
 * - DELETE 요청을 통해 특정 게시글을 즐겨찾기에서 제거
 * - 성공 시 204(No Content) 상태코드 반환
 * - 실패 시 false 반환 및 에러 로깅
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
