import defaultAxios from "@apis/common";

// 작성자 정보 타입
export interface Writer {
  email: string;
  name: string;
  profileUrl: string;
}

// 규정 항목 타입
export interface RegulationItem {
  title: string;
  content: string;
  itemList: RegulationItem[];
}

// 규정 응답 타입
export interface RegulationResponse {
  id: string;
  boardId: number;
  itemList: RegulationItem[];
}

// 게시글 상세 정보 응답 타입
export interface BoardDetailResponse {
  boardId: number;
  title: string;
  contents: string;
  level: number;
  category: string;
  fileUrl: string | null;
  createdAt: string;
  writer: Writer;
  isFavorite: boolean;
  regulationResponseDto: RegulationResponse | null;
  hasFile: boolean;
}

/**
 * 규정 상세 정보를 조회하는 API
 */
export const getBoardDetailAPI = async (
  boardId: number
): Promise<BoardDetailResponse> => {
  try {
    const response = await defaultAxios.get<BoardDetailResponse>(
      `/api/boards/${boardId}`
    );
    return response.data;
  } catch (error) {
    console.error("Error in getRegulationDetailAPI:", error);
    throw error;
  }
};

/**
 * 파일 다운로드 API
 */
export const downloadFileAPI = async (fileUrl: string): Promise<Blob> => {
  try {
    const response = await defaultAxios.get(`/api/files/download`, {
      params: { fileUrl },
      responseType: "blob",
    });
    return response.data;
  } catch (error) {
    console.error("Error in downloadFileAPI:", error);
    throw error;
  }
};

/**
 * 즐겨찾기 추가 API
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
 * 즐겨찾기 삭제 API
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
