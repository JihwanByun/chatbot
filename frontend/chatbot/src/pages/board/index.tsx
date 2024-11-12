import React, { useCallback, useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import InputSearch from "@components/atoms/input/InputSearch";
import Paging from "@components/atoms/paging/Paging";
import TableWithIconAndButton from "@components/atoms/table/TableWithIcon";
import { TableRowData } from "@components/atoms/table/TableWithIcon";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import {
  fetchFavoriteBoards,
  updateFavoriteStatus,
} from "@app/slices/favoriteBoardsSlice";
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
import { ButtonWithIconProps } from "@components/atoms/button/ButtonWithIcon";

/** 즐겨찾기 아이콘 경로 상수 */
export const favoritePath = "/src/assets/icons/favorite.svg"; // 즐겨찾기 되지 않은 상태 아이콘
export const favoritedPath = "/src/assets/icons/Favorited.svg"; // 즐겨찾기 된 상태 아이콘

/** 테이블 헤더 정의 */
const tableHeader = ["", "작성자", "제목", "등록일"];

const BoardListPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  // Redux store에서 상태 가져오기
  const { favorites, loading, hasMore, currentPage } = useAppSelector(
    (state) => state.favoriteBoards
  );

  // 로컬 상태 관리
  const [curPage, setCurPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [tableRows, setTableRows] = useState<TableRowData[]>([]);
  const [iconPaths, setIconPaths] = useState<{ [key: number]: string }>({});
  const [searchCondition, setSearchCondition] = useState<BoardSearchCondition>({
    level: undefined,
    name: undefined,
    title: undefined,
  });

  // 스타일 정의
  const footStyle =
    "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  // 페이지네이션 정보
  const pageInfo = useMemo<Pageable>(
    () => ({
      page: curPage - 1,
      size: 10,
      sort: ["createdAt,desc"],
    }),
    [curPage]
  );

  // 게시글 목록 조회
  const fetchBoards = useCallback(async () => {
    try {
      const cleanedCondition = Object.fromEntries(
        Object.entries(searchCondition).filter(
          ([_, value]) => value !== undefined && value !== ""
        )
      );

      const boardList: TableResponse | null = await getBoardListAPI(
        cleanedCondition,
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

  // 즐겨찾기 토글 처리
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
          const board = tableRows.find((row) => row.id === rowId);
          if (!board) return;
          dispatch(
            updateFavoriteStatus({
              boardId: rowId,
              isFavorite: !isFavorite,
              boardData: !isFavorite
                ? {
                    boardId: rowId,
                    title: board.columns[1],
                  }
                : null,
            })
          );

          // 아이콘 상태 즉시 업데이트
          setIconPaths((prev) => ({
            ...prev,
            [rowId]: !isFavorite ? favoritedPath : favoritePath,
          }));
        }
      } catch (error) {
        console.error("Error toggling favorite:", error);
      }
    },
    [iconPaths, tableRows, dispatch]
  );

  // 검색 처리
  const handleSearch = useCallback(
    (searchType: string, searchValue: string) => {
      const newCondition: BoardSearchCondition = {};
      switch (searchType) {
        case "작성자":
          newCondition.name = searchValue || undefined;
          break;
        case "제목":
          newCondition.title = searchValue || undefined;
          break;
      }
      setSearchCondition(newCondition);
      setCurPage(1);
    },
    []
  );

  // 더 많은 즐겨찾기 로드
  const handleLoadMore = () => {
    if (!loading && hasMore) {
      dispatch(fetchFavoriteBoards(currentPage + 1));
    }
  };

  // 페이지 변경 처리
  const handlePageChange = useCallback((page: number) => {
    setCurPage(page);
  }, []);

  // 컴포넌트 구성
  const footerComponents: ButtonWithIconProps[] = [
    {
      btnName: "채팅",
      styleName: footStyle,
      icon: "/src/assets/icons/chatting-icon.svg",
      handleClick: () => navigate("/chat"),
    },
    {
      btnName: "로그아웃",
      styleName: footStyle,
      icon: "/src/assets/icons/logout.svg",
      handleClick: () => {
        localStorage.removeItem("Authorization");
        navigate("/login");
      },
    },
  ];

  const sidebarComponents: ButtonWithIconProps[] = [
    {
      btnName: "규정목록",
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text.svg",
      handleClick: () => navigate("/board"),
    },
    ...favorites.map((favorite) => ({
      btnName: favorite.title,
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text.svg",
      handleClick: () => navigate(`/board/${favorite.boardId}`),
    })),
  ];

  // 초기 데이터 로딩
  useEffect(() => {
    if (currentPage === 0) {
      dispatch(fetchFavoriteBoards(0));
    }
  }, [dispatch, currentPage]);

  useEffect(() => {
    fetchBoards();
  }, [fetchBoards]);

  const boardListContent = (
    <>
      <div className="mb-4 flex justify-center mx-5">
        <InputSearch
          onIconClick={handleSearch}
          searchTypes={["제목", "작성자"]}
          placeholder="검색어를 입력하세요"
        />
      </div>

      <div className="mb-4">
        <TableWithIconAndButton
          columns={tableHeader}
          rows={tableRows}
          iconPaths={iconPaths}
          onIconClick={handleFavoriteToggle}
          onRowClick={(rowId) => navigate(`/board/${rowId}`)}
        />
      </div>

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

  return (
    <SembotLayout
      title="규정목록"
      sidebarComponents={sidebarComponents}
      footerComponents={footerComponents}
      hasMore={hasMore}
      onLoadMore={handleLoadMore}
      isLoading={loading}
      isRule={true}
    >
      {boardListContent}
    </SembotLayout>
  );
};

export default BoardListPage;
