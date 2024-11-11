import React, { useEffect, useState } from "react";
import SembotLayout from "../SembotLayout";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import TreeView from "@pages/board/TreeView";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import {
	setRevisionMode,
	startEditNode,
	saveNodeEdit,
	cancelEdit,
} from "@app/slices/treeSlice";

interface RegulationPageProps {
	title: string;
	userImage?: string;
	userName?: string;
	userEmail?: string;
	date?: string;
	boardTitle?: string;
	level?: string;
}

const RegulationPage: React.FC<RegulationPageProps> = ({
	userImage = "/src/assets/icons/user-profile-ex.svg",
	title = "규정 정보",
	userName = "작성자",
	userEmail = "semes@semes.com",
	date = "June 25, 2018, 3:26PM",
	boardTitle = "임직원 휴가 지침",
	level = "1",
}) => {
	const dispatch = useAppDispatch();
	const [isFavorited, setIsFavorited] = useState(false);
	const [role, setRole] = useState<string | null>("");

	const isRevisionMode = useAppSelector((state) => state.tree.isRevisionMode);
	const editNodeData = useAppSelector((state) => state.tree.editNodeData);

	useEffect(() => {
		const storedRole = localStorage.getItem("Role");
		if (storedRole) setRole(storedRole);
	}, []);

	const toggleFavorite = () => setIsFavorited(!isFavorited);
	const toggleRevisionMode = () => dispatch(setRevisionMode(!isRevisionMode));
	const discardEdit = () => dispatch(cancelEdit());

	// handleSaveEdit 함수 추가하여 saveNodeEdit에 필요한 데이터를 전달
	const handleSaveEdit = (nodeId: string) => {
		if (editNodeData) {
			dispatch(
				saveNodeEdit({
					id: nodeId,
					title: editNodeData.title || "",
					content: editNodeData.content || "",
				})
			);
		}
	};

	const getChildren = () => (
		<div className="bg-white rounded-lg px-6 space-y-6">
			<div className="flex items-center space-x-4">
				<ButtonOnlyIcon
					key="move-prev-board"
					icon="/src/assets/icons/go-to-prev.svg"
					styleName="p-2 hover:bg-gray-100 rounded"
				/>
				<ButtonOnlyIcon
					key="favorite"
					icon={
						isFavorited
							? "/src/assets/icons/favorited.svg"
							: "/src/assets/icons/favorite.svg"
					}
					width="20rem"
					styleName="p-2 hover:bg-gray-100 rounded"
					onClick={toggleFavorite}
				/>
				<ButtonOnlyIcon
					key="delete"
					icon="/src/assets/icons/delete.svg"
					width="20rem"
					styleName="p-2 hover:bg-gray-100 rounded"
				/>
				{(role === "관리자" && !isRevisionMode && (
					<ButtonOnlyIcon
						key="edit"
						icon="src/assets/icons/pen.svg"
						styleName={`p-2 hover:bg-gray-100 rounded ml-[16px] ${
							isRevisionMode ? "text-blue-600" : ""
						}`}
						width="20rem"
						onClick={toggleRevisionMode}
					/>
				)) || (
					<div className="flex">
						<ButtonOnlyIcon
							key="save"
							icon="src/assets/icons/save.svg"
							styleName={`p-2 hover:bg-gray-100 rounded ${
								isRevisionMode ? "text-blue-600" : ""
							}`}
							width="20rem"
							onClick={toggleRevisionMode}
						/>
						<ButtonOnlyIcon
							key="discard"
							icon="src/assets/icons/x-circle.svg"
							styleName={`p-2 hover:bg-gray-100 rounded ml-[8px] ${
								isRevisionMode ? "text-blue-600" : ""
							}`}
							width="20rem"
							onClick={discardEdit}
						/>
					</div>
				)}
			</div>

			<div className="flex">
				<div className="text-medium font-semibold">{boardTitle}</div>
				<span className="inline-flex items-center ml-4 px-2 py-1 rounded-full text-xs font-medium bg-gray-100">
					답변 레벨 : {level}
				</span>
			</div>

			<div className="flex items-center justify-between">
				<div className="flex items-center space-x-2">
					<img className="h-8 w-8 rounded-full" src={userImage} alt="" />
					<span className="font-medium font-semibold">{userName}</span>
					<span className="text-gray-500 text-xs">&lt;{userEmail}&gt;</span>
				</div>
				<div className="text-xs py-2 text-gray-500">{date}</div>
			</div>

			<div className="mt-4">
				<TreeView
					isRevisionMode={isRevisionMode}
					handleStartEdit={(node) =>
						dispatch(
							startEditNode({
								id: node.id,
								title: node.title || "", // title이 없을 때 빈 문자열로 설정
								content: node.content || "", // content가 없을 때 빈 문자열로 설정
							})
						)
					}
					handleSaveEdit={handleSaveEdit}
				/>
			</div>
		</div>
	);

	const sidebarComponents = [
		{
			btnName: "규정목록",
			styleName: "flex bg-blue-500 text-white py-2 px-4 rounded mx-1",
			icon: "/src/assets/icons/book-open-text.svg",
		},
		{
			btnName: "임직원 휴가 지침",
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
			title={title}
			sidebarComponents={sidebarComponents}
			footerComponents={footerComponents}
			children={getChildren()}
		/>
	);
};

export default RegulationPage;
