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
import { fetchFavoriteBoards, updateFavoriteStatus } from "@app/slices/favoriteBoardsSlice";
import { useAppDispatch, useAppSelector } from "@app/hooks";

/** 테이블 헤더 정의 */
const tableHeader = ["", "작성자", "제목", "등록일"];

/** 즐겨찾기 아이콘 경로 상수 */
export const favoritePath = "/src/assets/icons/favorite.svg"; // 즐겨찾기 되지 않은 상태 아이콘
export const favoritedPath = "/src/assets/icons/Favorited.svg"; // 즐겨찾기 된 상태 아이콘

/**
 * 게시판 목록을 표시
 * 검색, 페이징, 즐겨찾기 기능
 * @returns
 */
const BoardListContent: React.FC = () => {
  // 상태 관리
  const [curPage, setCurPage] = useState<number>(1); // 현재 페이지
  const [totalPages, setTotalPages] = useState<number>(1); // 전체 페이지
  const [tableRows, setTableRows] = useState<TableRowData[]>([]); // 테이블에 표시될 데이터
  const [searchCondition, setSearchCondition] = useState<BoardSearchCondition>({
    level: undefined, // 게시글 레벨
    name: undefined, // 작성자 이름
    title: undefined, // 게시글 제목
  });
  // Redux store에서 favorites 상태 가져오기
  const { favorites } = useAppSelector((state) => state.favoriteBoards);

  // Redux store에서 favorites 상태 가져오기

  // 페이지 이동을 위한 네비게이터
  const navigator = useNavigate();
  const dispatch = useAppDispatch();
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

  // 각 행의 아이콘 경로를 저장하는 상태
  const [iconPaths, setIconPaths] = useState<{ [key: number]: string }>({});

  const updateIconPaths = useCallback((boards: TableRowData[]) => {
    const newIconPaths: { [key: number]: string } = {};
    boards.forEach((row) => {
      const isFavorite = favorites.some(
        (favorite) => favorite.boardId === row.id
      );
      newIconPaths[row.id] = isFavorite ? favoritedPath : favoritePath;
    });
    setIconPaths(newIconPaths);
  }, [favorites]);

  const fetchBoards = useCallback(async () => {
    try {
      // 빈 값인 필드는 제외하고 API 요청
      const cleanedCondition = Object.fromEntries(
        Object.entries(searchCondition).filter(
          ([_, value]) => value !== undefined && value !== ""
        )
      );

      // API 호출 및 결과 처리
      const boardList: TableResponse | null = await getBoardListAPI(
        cleanedCondition,
        pageInfo
      );
      if (boardList) {
        setTableRows(boardList.contents); // 테이블 데이터 업데이트
        setIconPaths(boardList.iconPaths); // 즐겨찾기 아이콘 상태 업데이트
        setTotalPages(boardList.totalPages); // 전체 페이지 수 업데이트
      }
    } catch (error) {
      console.error("Error fetching boards:", error);
    }
  }, [searchCondition, pageInfo]);

  // useEffect(() => {
  //   if (tableRows.length > 0) {
  //     updateIconPaths(tableRows);
  //   }
  // }, [favorites, tableRows, updateIconPaths]);

  /**
   * 검색 조건 변경 처리 함수
   * - 검색어가 빈 문자열인 경우 undefined로 설정하여 전체 검색되도록 함
   * - 검색 시 첫 페이지로 이동
   */
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
      setCurPage(1); // 검색 시 첫 페이지로 이동
    },
    []
  );

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
          // 즐겨찾기 상태 업데이트
          const board = tableRows.find((row) => row.id === rowId);
         
          dispatch(
            updateFavoriteStatus({
              boardId: rowId,
              isFavorite: !isFavorite,
              boardData: !isFavorite ? board : null,
            })
          );
          // 목록 새로고침
          fetchBoards();
        }
      } catch (error) {
        console.error("Error toggling favorite:", error);
      }
    },
    [iconPaths, tableRows, dispatch, fetchBoards]
  );

  // 페이지 변경 처리
  const handlePageChange = useCallback((page: number) => {
    setCurPage(page);
  }, []);

  /**
   * 게시글 상세 페이지 이동 함수
   * - 게시글 ID를 URL 파라미터로 사용
   */
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
          searchTypes={["제목", "작성자"]}
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
