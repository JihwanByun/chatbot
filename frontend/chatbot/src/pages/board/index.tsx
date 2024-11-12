import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import BoardListContent from "./BoardListContent";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { fetchFavoriteBoards } from "@app/slices/favoriteBoardsSlice";
import { ButtonWithIconProps } from "@components/atoms/button/ButtonWithIcon";

const BoardListPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { favorites, loading, hasMore, currentPage } = useAppSelector(
    (state) => state.favoriteBoards
  );
  const footStyle =
    "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  // 즐겨찾기 목록 초기 로드
  useEffect(() => {
    if (currentPage === 0) {
      dispatch(fetchFavoriteBoards(0));
    }
  }, [dispatch, currentPage]);

  // 더 많은 즐겨찾기 로드
  const handleLoadMore = () => {
    if (!loading && hasMore) {
      dispatch(fetchFavoriteBoards(currentPage + 1));
    }
  };

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

  // 사이드바 컴포넌트 구성
  const sidebarComponents: ButtonWithIconProps[] = [
    {
      btnName: "규정목록",
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text.svg",
      handleClick: () => navigate("/board"),
    },
    // 즐겨찾기 목록을 버튼 컴포넌트로 변환
    ...favorites.map((favorite) => ({
      btnName: favorite.title,
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text.svg",
      handleClick: () => navigate(`/board/${favorite.boardId}`),
    })),
  ];

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
      <BoardListContent />
    </SembotLayout>
  );
};

export default BoardListPage;
