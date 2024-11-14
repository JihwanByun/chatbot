import React, { useState, useEffect } from "react";
import { useNavigate, useLocation, Location } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import Analytics from "./analystics";
import MemberManagement from "./MemberManagement";
import FeedbackManagement from "./FeedbackManagement";
import CategoryManagement from "./CategoryManagement";

interface LocationState {
  activeTab?: string;
}

const SembotPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation() as Location & { state: LocationState };

  const [activeButton, setActiveButton] = useState<string>(() => {
    // 초기 상태 설정 시 location.state 확인
    return location.state?.activeTab || "회원 관리";
  });

  // location.state가 변경될 때마다 activeButton 업데이트
  useEffect(() => {
    if (location.state?.activeTab) {
      setActiveButton(location.state.activeTab);
    }
  }, [location.state?.activeTab]);

  const handleClick = (btnName: string) => {
    setActiveButton(btnName);
  };

  const handleClickRegulation = () => {
    navigate("/board");
    setActiveButton("");
  };

  const SidebarButtons = [
    {
      styleName: "flex py-2 px-4 rounded mx-1",
      btnName: "회원 관리",
      icon: "/src/assets/icons/user-manage.svg",
      handleClick: () => handleClick("회원 관리"),
      isActive: activeButton === "회원 관리",
    },
    {
      styleName: "flex py-2 px-4 rounded mx-1",
      btnName: "피드백 관리",
      icon: "/src/assets/icons/like.svg",
      handleClick: () => handleClick("피드백 관리"),
      isActive: activeButton === "피드백 관리",
    },
    {
      styleName: "flex py-2 px-4 rounded mx-1",
      btnName: "카테고리 관리",
      icon: "/src/assets/icons/list.svg",
      handleClick: () => handleClick("카테고리 관리"),
      isActive: activeButton === "카테고리 관리",
    },
    {
      styleName: "flex py-2 px-4 rounded mx-1",
      btnName: "통계",
      icon: "/src/assets/icons/trending-up.svg",
      handleClick: () => handleClick("통계"),
      isActive: activeButton === "통계",
    },
  ];

  const FooterButtons = [
    {
      styleName: "flex py-2 px-4 rounded mx-1",
      btnName: "규정 확인하기",
      icon: "/src/assets/icons/book-open-text-footer.svg",
      isFooter: true,
      handleClick: handleClickRegulation,
    },
    {
      styleName: "flex py-2 px-4 rounded mx-1",
      btnName: "로그아웃",
      icon: "/src/assets/icons/logout.svg",
      isFooter: true,
    },
  ];

  const getChildren = () => {
    switch (activeButton) {
      case "통계":
        return <Analytics />;
      case "회원 관리":
        return <MemberManagement />;
      case "피드백 관리":
        return <FeedbackManagement />;
      case "카테고리 관리":
        return <CategoryManagement />;
      default:
        return <MemberManagement />;
    }
  };

  return (
    <SembotLayout
      title={activeButton}
      isRule={false}
      sidebarComponents={SidebarButtons}
      footerComponents={FooterButtons}
      children={getChildren()}
    />
  );
};

export default SembotPage;
