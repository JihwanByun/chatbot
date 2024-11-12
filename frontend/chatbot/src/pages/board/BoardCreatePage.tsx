import React from "react";
import { useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import TreeCreate from "./TreeCreate"; // TreeCreate 컴포넌트 import

const BoardCreatePage: React.FC = () => {
	const navigate = useNavigate();

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
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
		},
	];

	const footerComponents = [
		{
			btnName: "채팅",
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/chatting-icon.svg",
		},
		{
			btnName: "규정 확인하기",
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
		},
		{
			btnName: "로그아웃",
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
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
