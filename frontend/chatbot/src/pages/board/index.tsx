import React, { useCallback, useEffect, useState, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import InputSearch from "@components/atoms/input/InputSearch";
import Paging from "@components/atoms/paging/Paging";
import TableWithIconAndButton from "@components/atoms/table/TableWithIcon";
import { TableRowData } from "@components/atoms/table/TableWithIcon";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { fetchFavoriteBoards, updateFavoriteStatus } from "@app/slices/favoriteBoardsSlice";
import { getBoardListAPI, TableResponse, BoardSearchCondition, Pageable } from "@apis/board/boardApi";
import { createFavoriteAPI, deleteFavoriteAPI } from "@apis/board/boardFavoriteApi";
import { ButtonProps as ButtonWithIconProps } from "@components/atoms/button/ButtonWithIcon";
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import dayjs from "dayjs";
import { UserRole } from "@util/userConfig";
import { getNavigationConfig } from "@pages/admin/adminNavigation";
import { errorAlert } from "@util/alert";

/** 즐겨찾기 아이콘 경로 상수 */
export const favoritePath = "/src/assets/icons/favorite.svg"; // 즐겨찾기 되지 않은 상태 아이콘
export const favoritedPath = "/src/assets/icons/Favorited.svg"; // 즐겨찾기 된 상태 아이콘

/** 테이블 헤더 정의 */
const tableHeader = ["", "작성자", "제목", "등록일"];

const BoardListPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  // Redux store에서 상태 가져오기
  const { favorites, loading, hasMore, currentPage} = useAppSelector((state) => state.favoriteBoards);
  const favoriteError = useAppSelector((state) => state.favoriteBoards.error);

  // 로컬 상태 관리
  const [error, setError] = useState<string | null>(null);
  const [curPage, setCurPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [tableRows, setTableRows] = useState<TableRowData[]>([]);
  const [iconPaths, setIconPaths] = useState<{ [key: number]: string }>({});
  const [searchCondition, setSearchCondition] = useState<BoardSearchCondition>({
    level: undefined,
    name: undefined,
    title: undefined,
  });
  const { role } = useAppSelector((state) => state.users);

  // 스타일 정의
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  /**
   * 페이지네이션 정보 설정
   * - page: API 요청 시 0부터 시작하므로 현재 페이지에서 1을 뺌
   * - size: 한 페이지당 표시할 항목 수 (default : 10)
   * - sort: 정렬 기준 (생성일 기준 내림차순)
   */
  const pageInfo = useMemo<Pageable>(
    () => ({
      page: curPage - 1,
      size: 10,
      sort: ["createdAt,desc"],
    }),
    [curPage]
  );

	const handleError = (error : unknown) => {
		if (error instanceof Error) {
			errorAlert(error);
      setError(error.message);
		}else{
      setError("알 수 없는 오류가 발생했습니다.");
		}
	}

  // 게시글 목록 조회
  const fetchBoards = useCallback(async () => {
    try {
      const boardList: TableResponse | null = await getBoardListAPI(searchCondition, pageInfo);
      if (boardList == null) return;
      console.log("!!!!!");
      
      // 날짜 포맷팅이 적용된 데이터로 변환
      const formattedRows = boardList.contents.map((row) => ({
        ...row,
        columns: [
          row.columns[0], // 첫 번째 열
          row.columns[1], // 두 번째 열
          formatDate(row.columns[2]), // 세 번째 열 (날짜)
        ],
      }));

      setTableRows(formattedRows);
      setIconPaths(boardList.iconPaths);
      setTotalPages(boardList.totalPages);
    } catch (error) {
      handleError(error);
    } 
  }, [searchCondition, pageInfo]);

  /**
   * 즐겨찾기 토글 처리 함수
   * - 현재 즐겨찾기 상태를 확인하고 반대 상태로 변경
   * - API 호출 성공 시 목록을 다시 불러와 화면 갱신
   */
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

  /**
   * 검색 조건 변경 처리 함수
   * - 검색어가 빈 문자열인 경우 undefined로 설정하여 전체 검색되도록 함
   * - 검색 시 첫 페이지로 이동
   */
  const handleSearch = useCallback((searchType: string, searchValue: string) => {
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
    setCurPage(1); // 검색 시 첫 페이지로 이동
  }, []);

  // 더 많은 즐겨찾기 로드
  const handleLoadMore = useCallback(() => {
    if (!favoriteError && !loading && hasMore) {
      dispatch(fetchFavoriteBoards(currentPage + 1));
    }
  },[favoriteError, loading, hasMore, currentPage, dispatch]);

  // 페이지 변경 처리
  const handlePageChange = useCallback((page: number) => {
    setCurPage(page);
  }, []);

  const handleClickWrite = () => {
    navigate("/treecreate");
  };

  // 컴포넌트 구성
  const { footerComponents } = getNavigationConfig(role, navigate, dispatch);

  const sidebarComponents: ButtonWithIconProps[] = [
    {
      btnName: "규정 목록",
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate("/board"),
    },
    ...favorites.map((favorite) => ({
      btnName: favorite.title,
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate(`/board/${favorite.boardId}`),
    })),
  ];

  // 초기 데이터 로딩
  useEffect(() => {
    if (!favoriteError && currentPage === 0) {
      try{
        dispatch(fetchFavoriteBoards(0));
      }catch(error){
        handleError(error);
      }
    }
  }, [favoriteError, dispatch, currentPage]);

  useEffect(() => {
    if(!error){
      fetchBoards();
    }
    
  }, [error, location.state?.refresh, fetchBoards, navigate, location.pathname]);

  // 날짜 포맷팅 함수
  const formatDate = (dateString: string): string => {
    const today = dayjs();
    const targetDate = dayjs(dateString);

    if (targetDate.isSame(today, "day")) {
      return targetDate.format("오늘 HH:mm");
    }

    if (targetDate.isSame(today, "year")) {
      return targetDate.format("MM/DD HH:mm");
    }

    return targetDate.format("YY/MM/DD");
  };

  const boardListContent = (
    <>
      {/* 검색 입력란 */}
      <div className="mb-4 flex justify-center mx-5">
        <InputSearch onIconClick={handleSearch} searchTypes={["제목", "작성자"]} placeholder="검색어를 입력하세요" />
      </div>

      {/* 게시글 영역 */}
      <div className="mb-4 px-5">
        <TableWithIconAndButton
          columns={tableHeader}
          rows={tableRows}
          iconPaths={iconPaths}
          onIconClick={handleFavoriteToggle}
          onRowClick={(rowId) => navigate(`/board/${rowId}`)}
        />
      </div>

      {/* 페이지 번호 영역 */}
      <div className="flex justify-center">
        <div className="absolute bottom-5 mt-4">
          <Paging curPage={curPage} totalPage={totalPages} onPageChange={handlePageChange} />
        </div>
      </div>

      {/*글 쓰기 버튼*/}
      {role === UserRole.USER_WRITE ? (
        <div className="absolute top-5 right-[150px]">
          <ButtonPrimary btnName="규정 글 쓰기" styleName="bg-blue-100" handleClick={handleClickWrite} />
        </div>
      ) : (
        <></>
      )}
    </>
  );

  return (
    <SembotLayout
      title="규정 목록"
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
