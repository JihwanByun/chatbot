import React, { useEffect, useState } from "react";
import SembotLayout from "../SembotLayout";
import BoardListContent from "./BoardListContent";

const footerComponents = [
	{
		btnName: "채팅",
		styleName: "flex text-white py-2 px-4 rounded mx-1",
		icon: "/src/assets/icons/chatting-icon.svg",
	},
	{
		btnName: "규정 확인하기",
		styleName: "flex text-white py-2 px-4 rounded mx-1",
		icon: "/src/assets/icons/book-open-text.svg",
	},
	{
		btnName: "로그아웃",
		styleName: "flex text-white py-2 px-4 rounded mx-1",
		icon: "/src/assets/icons/logout.svg",
	},
];

const BoardListPage: React.FC = () => {
	const sidebarComponents = [
		{
			btnName: "규정목록",
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
		},
		{
			btnName: "즐겨찾는 규정 1",
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
		},
	];

	return (
		<>
			<SembotLayout
				title={"규정목록"}
				sidebarComponents={sidebarComponents}
				footerComponents={footerComponents}
			>
				<BoardListContent />
			</SembotLayout>
		</>
	);
};

export default BoardListPage;
