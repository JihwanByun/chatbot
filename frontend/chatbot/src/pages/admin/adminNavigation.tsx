import { UserRole } from "@util/userConfig";
import { NavigateFunction } from "react-router-dom";
import { AppDispatch } from "@app/store";
import { logoutUser } from "@app/slices/userSlice";

interface NavConfig {
  footerComponents: {
    btnName: string;
    styleName: string;
    icon: string;
    handleClick: () => void;
  }[];
}

const footStyle =
  "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";

export const getNavigationConfig = (role: UserRole, navigate: NavigateFunction, dispatch: AppDispatch): NavConfig => {
  if (role === UserRole.ADMIN) {
    return {
      footerComponents: [
        {
          btnName: "회원 관리",
          styleName: footStyle,
          icon: "/src/assets/icons/user-manage-white.svg",
          handleClick: () => navigate("/adminPage", { state: { activeTab: "회원 관리" } }),
        },
        {
          btnName: "피드백 관리",
          styleName: footStyle,
          icon: "/src/assets/icons/like-white.svg",
          handleClick: () => navigate("/adminPage", { state: { activeTab: "피드백 관리" } }),
        },
        {
          btnName: "카테고리 관리",
          styleName: footStyle,
          icon: "/src/assets/icons/list-white.svg",
          handleClick: () => navigate("/adminPage", { state: { activeTab: "카테고리 관리" } }),
        },
        {
          btnName: "통계",
          styleName: footStyle,
          icon: "/src/assets/icons/trending-up-white.svg",
          handleClick: () => navigate("/adminPage", { state: { activeTab: "통계" } }),
        },
        {
          btnName: "로그아웃",
          styleName: footStyle,
          icon: "/src/assets/icons/logout.svg",
          handleClick: async () => {
            await dispatch(logoutUser());
            navigate("/");
          },
        },
      ],
    };
  }

  return {
    footerComponents: [
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
        handleClick: async () => {
          await dispatch(logoutUser());
          navigate("/");
        },
      },
    ],
  };
};
