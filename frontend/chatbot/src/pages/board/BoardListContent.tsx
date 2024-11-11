import { getBoardListAPI } from "@apis/board/boardApi";
import InputSearch from "@components/atoms/input/InputSearch";
import Paging from "@components/atoms/paging/Paging";
import TableWithIconAndButton from "@components/atoms/table/TableWithIcon";
import { TableRowData } from "@components/atoms/table/TableWithIcon";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { TableResponse } from "@apis/board/boardApi";
import { createFavoriteAPI } from "@apis/board/boardFavoriteApi";
import { deleteFavoriteAPI } from "@apis/board/boardFavoriteApi";

const tableHeader = ["", "작성자", "제목", "등록일"];

const totalPage = 100;
export const favoritePath = "/src/assets/icons/favorite.svg";
export const favoritedPath = "/src/assets/icons/Favorited.svg";

const BoardListContent: React.FC = () => {
  const [curPage, setCurPage] = useState(1);
  const [tableRows, setTableRows] = useState<TableRowData[]>([]);

  const navigator = useNavigate();
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

  const routeBoardDetail = (rowId: number) => {
    navigator(`/board/${rowId}`);
  };

  const fetchBoards = useCallback(async () => {
    const boardList: TableResponse | null = await getBoardListAPI();
    if (boardList == null) {
      return;
    }
    console.log(boardList);

    setTableRows(boardList.contents);
    setIconPaths(boardList.iconPaths);
  }, [tableRows, iconPaths]);

  useEffect(() => {
    // 게시판 목록 조회
    fetchBoards();
  }, []);

  // 즐겨찾기 토글 처리 함수
  const handleFavoriteToggle = async (rowId: number) => {
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
        // API 호출이 성공하면 아이콘 상태를 업데이트
        setIconPaths((currentIconPaths) => ({
          ...currentIconPaths,
          [rowId]: isFavorite ? favoritePath : favoritedPath,
        }));

        // 목록을 다시 불러와 최신 상태 반영
        fetchBoards();
      }
    } catch (error) {
      console.error("Error toggling favorite:", error);
    }
  };

  return (
    <>
      {/* 검색 입력란 */}
      <div className="mb-4 flex justify-center mx-5">
        <InputSearch />
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
        {/* 게시글 내용이 여기에 표시될 수 있습니다 */}
      </div>

      {/* 페이지 번호 영역 */}
      <div className="flex justify-center">
        <div className="absolute bottom-5 mt-4">
          <Paging
            curPage={curPage}
            totalPage={totalPage}
            onPageChange={(page: number) => {
              setCurPage(page);
            }}
          />
        </div>
      </div>
    </>
  );
};

export default BoardListContent;
