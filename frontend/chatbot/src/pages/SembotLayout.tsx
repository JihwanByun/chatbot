import React, { useState } from "react";
import Header from "@components/atoms/header/Header";
import Sidebar from "@components/atoms/sidebar/Sidebar";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";

export interface SembotLayoutProps {
  children?: React.ReactNode;
  title?: string;
  isRule?: boolean;
  hideHeader?: boolean;
  sidebarComponents?: React.ComponentProps<typeof ButtonWithIcon>[];
  footerComponents?: React.ComponentProps<typeof ButtonWithIcon>[];
  hasMore?: boolean;
  onLoadMore?: () => void;
  isLoading?: boolean;
}

const SembotLayout: React.FC<SembotLayoutProps> = ({
  children,
  title,
  isRule = true,
  hideHeader = false,
  sidebarComponents = [
    {
      btnName: "규정목록",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/plus.svg",
    },
    {
      btnName: "즐겨찾는 규정 1",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/plus.svg",
    },
  ],
  footerComponents = [
    {
      btnName: "footer Click Me 1",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/plus.svg",
    },
    {
      btnName: "footer Click Me 2",
      styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
      icon: "/src/assets/icons/plus.svg",
    },
  ],
  hasMore,
  onLoadMore,
  isLoading,
}) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="flex w-full h-full text-sm">
      {/* 모바일 메뉴 토글 버튼 */}
      <button
        className={`fixed top-4 left-4 z-50 sm:hidden
          transition-all duration-200 ease-in-out
          ${isSidebarOpen 
            ? 'opacity-0 pointer-events-none' 
            : 'opacity-100'
          }  
        `}
        onClick={toggleSidebar}
      >
        {/* 햄버거 메뉴 아이콘 */}
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>

      {/* 오버레이 (모바일에서 사이드바가 열렸을 때) */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 sm:hidden"
          onClick={toggleSidebar}
        />
      )}

      {/* 좌측 Sidebar 영역 */}
      <div
        className={`
          sm:w-[16rem] w-64 h-full fixed left-0 top-0 
          transition-transform duration-100 ease-in-out
          sm:translate-x-0 
          ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          sm:block
          bg-white 
          z-40
        `}
      >
        <Sidebar
          components={sidebarComponents}
          footerComponents={footerComponents}
          isRule={isRule}
          hasMore={hasMore}
          onLoadMore={onLoadMore}
          isLoading={isLoading}
        />
      </div>
      
      {/* 우측 콘텐츠 영역 */}
      <div className="absolute sm:left-[16rem] left-0 h-full right-0">
        {/* Header 영역 */}
        <div className="my-4 mx-2" hidden={hideHeader}>
          <Header title={title} />
        </div>
        {children}
      </div>
    </div>
  );
};

export default SembotLayout;