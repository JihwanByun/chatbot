import React, { useEffect, useState } from "react";
import SembotLayout from "../SembotLayout";
import BoardListContent from "./BoardListContent";

const footerComponents = [
	{
		btnName: "채팅",
		styleName: "flex text-white py-2 px-4 rounded mx-1",
		icon: "/src/assets/icons/chatting-icon.svg",
		isFooter: true,
	},
	{
		btnName: "규정 확인하기",
		styleName: "flex text-white py-2 px-4 rounded mx-1",
		icon: "/src/assets/icons/book-open-text-footer.svg",
		isFooter: true,
	},
	{
		btnName: "로그아웃",
		styleName: "flex text-white py-2 px-4 rounded mx-1",
		icon: "/src/assets/icons/logout.svg",
		isFooter: true,
	},
];

const BoardListPage: React.FC = () => {
	const sidebarComponents = [
		{
			btnName: "규정목록",
			styleName: "flex py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
			isActive: true,
		},
		{
			btnName: "즐겨찾는 규정 1",
			styleName: "flex py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
			isActive: true,
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
