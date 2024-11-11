import InputSearch from "@components/atoms/input/InputSearch";
import Paging from "@components/atoms/paging/Paging";
import TableWithIconAndButton from "@components/atoms/table/TableWithIcon";
import { TableRowData } from "@components/atoms/table/TableWithIcon";
import React, { useCallback, useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  getBoardListAPI,
  TableResponse,
  BoardSearchCondition,
  Pageable,
} from "@apis/board/boardApi";
import {
  createFavoriteAPI,
  deleteFavoriteAPI,
} from "@apis/board/boardFavoriteApi";

const tableHeader = ["", "작성자", "제목", "등록일"];

export const favoritePath = "/src/assets/icons/favorite.svg";
export const favoritedPath = "/src/assets/icons/Favorited.svg";

const BoardListContent: React.FC = () => {
  const [curPage, setCurPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [tableRows, setTableRows] = useState<TableRowData[]>([]);
  const [searchCondition, setSearchCondition] = useState<BoardSearchCondition>(
    {}
  );

  const navigator = useNavigate();

  // 페이지 정보
  const pageInfo = useMemo<Pageable>(
    () => ({
      page: curPage - 1,
      size: 10,
      sort: ["createdAt,desc"],
    }),
    [curPage]
  );

  // 각 행의 아이콘 경로를 저장하는 상태
  const [iconPaths, setIconPaths] = useState<{ [key: number]: string }>(
    tableRows.reduce((acc, row) => ({ ...acc, [row.id]: favoritePath }), {})
  );

  // 아이콘 클릭 시 해당 행의 아이콘 경로만 토글하는 함수
  const clickEvent = (rowId: number) => {
    setIconPaths((currentIconPaths) => ({
      ...currentIconPaths,
      [rowId]:
        currentIconPaths[rowId] === favoritePath ? favoritedPath : favoritePath,
    }));
  };

  const fetchBoards = useCallback(async () => {
    try {
      const boardList: TableResponse | null = await getBoardListAPI(
        searchCondition,
        pageInfo
      );
      if (boardList) {
        setTableRows(boardList.contents);
        setIconPaths(boardList.iconPaths);
        setTotalPages(boardList.totalPages);
      }
    } catch (error) {
      console.error("Error fetching boards:", error);
    }
  }, [searchCondition, pageInfo]);

  // 검색 조건 변경 처리
  const handleSearch = useCallback(
    (searchType: string, searchValue: string) => {
      const newCondition: BoardSearchCondition = {};

      switch (searchType) {
        case "작성자":
          newCondition.name = searchValue;
          break;
        case "제목":
          newCondition.title = searchValue;
          break;
      }

      setSearchCondition(newCondition);
      setCurPage(1); // 검색 시 첫 페이지로 이동
    },
    []
  );

  // 즐겨찾기 토글 처리 함수
  const handleFavoriteToggle = useCallback(
    async (rowId: number) => {
      const currentPath = iconPaths[rowId];
      const isFavorite = currentPath === favoritedPath;

      try {
        let success;
        if (isFavorite) {
          success = await deleteFavoriteAPI(rowId);
        } else {
          success = await createFavoriteAPI(rowId);
        }

        if (success) {
          // 성공 시에만 목록 갱신
          fetchBoards();
        }
      } catch (error) {
        console.error("Error toggling favorite:", error);
      }
    },
    [iconPaths, fetchBoards]
  );

  // 페이지 변경 처리
  const handlePageChange = useCallback((page: number) => {
    setCurPage(page);
  }, []);

  // 게시글 상세 페이지 이동
  const routeBoardDetail = useCallback(
    (rowId: number) => {
      navigator(`/board/${rowId}`);
    },
    [navigator]
  );

  // 초기 데이터 로딩 및 검색/페이지 변경 시 데이터 갱신
  useEffect(() => {
    fetchBoards();
  }, [fetchBoards]);

  return (
    <>
      {/* 검색 입력란 */}
      <div className="mb-4 flex justify-center mx-5">
        <InputSearch
          onIconClick={handleSearch}
          searchTypes={["작성자", "제목"]}
          placeholder="검색어를 입력하세요"
        />
      </div>

      {/* 게시글 영역 */}
      <div className="mb-4">
        <TableWithIconAndButton
          columns={tableHeader}
          rows={tableRows}
          iconPaths={iconPaths}
          onIconClick={handleFavoriteToggle}
          onRowClick={routeBoardDetail}
        />
      </div>

      {/* 페이지 번호 영역 */}
      <div className="flex justify-center">
        <div className="absolute bottom-5 mt-4">
          <Paging
            curPage={curPage}
            totalPage={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      </div>
    </>
  );
};

export default BoardListContent;
