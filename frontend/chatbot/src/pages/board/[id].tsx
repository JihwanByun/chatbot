import React, { useState } from "react";
import SembotLayout from "../SembotLayout";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";

interface RegulationPageProps {
	/** 전체 제목 */
	title: string;
	/** 작성자 이미지 URL */
	userImage?: string;
	/** 작성자 이름 */
	userName?: string;
	/** 작성자 사번 */
	userNumber?: string;
	/** PDF URL */
	pdfUrl?: string;
	/** 작성자 이메일 */
	userEmail?: string;
	/** 게시글 등록 날짜 */
	date?: string;
	/** 게시글 제목 */
	boardTitle?: string;
	/** 게시글 접근 레벨 */
	level?: string;
	/** 게시글 요약 */
	summary?: string;
}

const RegulationPage: React.FC<RegulationPageProps> = ({
	userImage = "/src/assets/icons/user-profile-ex.svg",
	pdfUrl = "/src/assets/icons/user-profile-ex.svg",
	title = "규정 정보",
	userName = "작성자",
	userEmail = "semes@semes.com",
	date = "June 25, 2018, 3:26PM",
	boardTitle = "임직원 휴가 지침",
	level = "1",
	summary = "업로드 된 pdf or hwp에 대한 간단 요약입니다.",
}) => {
	const [isFavorited, setIsFavorited] = useState(false);

	// 즐겨찾기 여부에 따른 아이콘 표시
	const toggleFavorite = () => {
		setIsFavorited(!isFavorited);
	};

	// PDF 다운로드 처리 함수
	const handleDownloadPDF = async () => {
		try {
			// fetch를 사용하여 PDF 파일 가져오기
			const response = await fetch(pdfUrl);
			const blob = await response.blob();

			// 다운로드 링크 생성
			const downloadUrl = window.URL.createObjectURL(blob);
			const link = document.createElement("a");
			link.href = downloadUrl;
			// TODO
			// svg를 pdf로 변환
			link.download = "규정문서.svg"; // 다운로드될 파일 이름

			// 다운로드 트리거
			document.body.appendChild(link);
			link.click();
			document.body.removeChild(link);

			// URL 정리
			window.URL.revokeObjectURL(downloadUrl);
		} catch (error) {
			console.error("PDF 다운로드 중 오류 발생:", error);
			alert("PDF 다운로드에 실패했습니다.");
		}
	};

	const getChildren = () => (
		<div className="bg-white rounded-lg px-6 space-y-6">
			{/* 게시판 상단 버튼 */}
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
			</div>

			<div className="flex">
				<div className="text-medium font-semibold">{boardTitle}</div>
				{/* 레벨 태그 */}
				<span className="inline-flex items-center ml-4 px-2 py-1 rounded-full text-xs font-medium bg-gray-100">
					답변 레벨 : {level}
				</span>
			</div>

			{/* 컨텐츠 란 */}
			<div className="space-y-4">
				<div className="flex items-center justify-between">
					<div className="flex items-center space-x-2">
						<img className="h-8 w-8 rounded-full" src={userImage} alt="" />
						<span className="font-medium font-semibold">{userName}</span>
						<span className="text-gray-500 text-xs">&lt;{userEmail}&gt;</span>
					</div>
					<div>
						<div className="text-xs py-2 text-gray-500">{date}</div>

						{/* 파일 표시 및 다운로드 */}
						<div
							className="flex justify-end space-x-2 text-sm text-blue-600 pt-2 py-2 cursor-pointer hover:text-blue-800"
							onClick={handleDownloadPDF}
						>
							<ButtonWithIcon
								btnName="파일 다운로드"
								styleName="items-center text-center text-sm flex rounded"
								icon="/src/assets/icons/document-download.svg"
							/>
						</div>
					</div>
				</div>

				<div className="text-sm text-gray-700">{summary}</div>
			</div>
			<hr />

			<div className="flex space-x-2 bottom-2">
				<ButtonWithIcon
					btnName="이전 게시글 보기"
					styleName="text-xs flex py-2 px-4 rounded space-x-2 rounded-md border-solid border-gray-700 border"
					icon="/src/assets/icons/prev-board.svg"
				/>
				<ButtonWithIcon
					btnName="다음 게시글 보기"
					styleName="text-xs flex py-2 px-4 rounded space-x-2 rounded-md border-solid border-gray-700 border"
					icon="/src/assets/icons/next-board.svg"
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
