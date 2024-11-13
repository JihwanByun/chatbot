import React from "react";
import { useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import TreeCreate from "./TreeCreate"; // TreeCreate 컴포넌트 import
import { useAppSelector } from "@app/hooks";

const BoardCreatePage: React.FC = () => {
  // 스타일 정의
  const footStyle =
    "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  const navigate = useNavigate();
  // Redux store에서 상태 가져오기
  const { favorites } = useAppSelector((state) => state.favoriteBoards);
  const getChildren = () => (
    <div className="bg-white rounded-lg px-6 space-y-6 text-left">
      {/* 상단 버튼 영역 */}
      <div className="flex items-center space-x-4">
        <ButtonOnlyIcon
          key="move-prev-board"
          icon="/src/assets/icons/go-to-prev.svg"
          styleName="p-2 hover:bg-gray-100 rounded"
          onClick={() => navigate(-1)}
        />
      </div>

      {/* TreeCreate 컴포넌트 */}
      <TreeCreate />
    </div>
  );

  const sidebarComponents = [
    {
      btnName: "규정목록",
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
    },
    ...favorites.map((favorite) => ({
      btnName: favorite.title,
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate(`/board/${favorite.boardId}`),
    })),
  ];

  const footerComponents = [
    {
      btnName: "채팅",
      styleName: footStyle,
      icon: "/src/assets/icons/chatting-icon.svg",
    },
    {
      btnName: "로그아웃",
      styleName: footStyle,
      icon: "/src/assets/icons/logout.svg",
    },
  ];

  return (
    <SembotLayout
      title="규정 작성"
      sidebarComponents={sidebarComponents}
      footerComponents={footerComponents}
      children={getChildren()}
    />
  );
};

export default BoardCreatePage;
