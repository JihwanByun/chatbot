import defaultAxios from "@apis/common";
import { TableRowData } from "@components/atoms/table/TableWithIcon";
import { favoritePath, favoritedPath } from "@pages/board/BoardListContent";

//===== Backend API =====//
// Type 지정
// Board 목록 조회 시 처음 값
type BoardListResponse = {
  contents: BoardResponse[];
  page: number;
  size: number;
  totalPages: number;
  totalElements: number;
};

type BoardResponse = {
  boardId: number;
  title: string;
  contents: string;
  createdAt: string;
  name: string;
  isFavorite: boolean;
};

type BoardSearchCondition = {
  level?: number;
  name?: string;
  title?: string;
};

type Pageable = {
  page: number;
  size: number;
  sort: string[];
};

export type TableResponse = {
  contents: TableRowData[];
  iconPaths: { [key: number]: string };
  page: number;
  size: number;
  totalPages: number;
  totalElements: number;
};

const convertToTable = (res: BoardListResponse): TableResponse => {
  return {
    ...res,
    contents: convertToTableRowList(res.contents),
    iconPaths: res.contents.reduce(
      (acc, content) => ({
        ...acc,
        [content.boardId]: content.isFavorite ? favoritedPath : favoritePath,
      }),
      {}
    ),
  };
};

const convertToTableRow = (res: BoardResponse): TableRowData => {
  return {
    id: res.boardId,
    columns: [res.name, res.title, res.createdAt],
  };
};

const convertToTableRowList = (response: BoardResponse[]): TableRowData[] => {
  return response.map(convertToTableRow);
};

export const getBoardListAPI = async (
  condition?: BoardSearchCondition,
  pageInfo?: Pageable
): Promise<TableResponse | null> => {
  return defaultAxios
    .get<BoardListResponse>("/api/boards", {
      params: {
        ...condition,
        ...pageInfo,
      },
    })
    .then((res) => {
      return convertToTable(res.data);
    })
    .catch((error) => {
      console.error("Error in boardListAPI:", error);
      return null;
    });
};
